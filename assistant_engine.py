"""
Headless assistant engine: route user message, run tools, call LLM, return reply string.
Used by both the desktop GUI and the FastAPI server.
"""

import json
import os
import subprocess
import threading
from typing import Any, Callable, Optional

from test import ask_lmstudio
from lights_client import (
    get_lights_state,
    toggle_all_lights,
    LightsClientError,
    set_lights_auto,
)
from gmail_client import search_gmail, GmailClientError

# Plex sync (same as chat_gui; override with PLEX_SYNC_DIR env if needed)
PLEX_SYNC_DIR = os.environ.get("PLEX_SYNC_DIR", r"H:\Coding\Python Projects\plex_sync")
PLEX_SYNC_PY = os.path.join(PLEX_SYNC_DIR, ".venv", "Scripts", "python.exe")
PLEX_SYNC_MAIN = os.path.join(PLEX_SYNC_DIR, "main.py")

# Gmail broad-term filter for list searches
_GENERIC_BROAD_TERMS = {"review", "feature", "contract"}


def _default_log(_msg: str) -> None:
    """No-op log when none provided."""
    pass


class AssistantEngine:
    """
    Stateless routing + tools + one LLM call per message.
    Optionally keeps last_route / last_user_message for follow-up handling.
    """

    def __init__(self, log_fn: Optional[Callable[[str], None]] = None):
        self.log = log_fn or _default_log
        self.last_route: dict | None = None
        self.last_user_message: str | None = None

    def route(self, user_text: str) -> dict:
        """Return {action, params} for the user message (LLM + heuristic fallback)."""
        prev_msg = self.last_user_message or ""
        prev_route_json = json.dumps(self.last_route) if self.last_route is not None else "null"

        routing_prompt = f"""
You are a command router for a desktop assistant called Galadrial.
Your job is to look at the user's message and decide which ACTION to take.

Valid actions:
- "none"  -> no side effects, just chat with the user.
- "lights.set_state" -> control my Govee lights.
    - params:
        - state: "on" | "off" | "auto"
- "lights.get_state" -> check whether the lights are currently on or off (no params).
- "plex_sync.run" -> run the user's Plex sync app (syncs media to a server). No params. Runs in the background; the app will tell the user when it finishes.
- "gmail.search" -> search my Gmail inbox via IMAP.
    - params:
        - query: string   (the search terms to send to Gmail—YOU interpret the user's intent)
        - scope: "unread" | "all"
        - result_type: "count" | "list"
        - category: one of "updates", "primary", "promotions", "social", "forums" (optional)
        - broad_search_terms: optional array of 2–6 short terms for "list" searches. When the user
          is asking to find/look through emails (e.g. "last email about X", "any acceptance from
          a literary mag"), set this to words that might appear in such emails so we cast a wider
          net (e.g. ["submission", "accepted", "acceptance", "literary", "magazine"]). Omit for
          count-only or when the user's query is already narrow.

Rules:
- If the user is clearly asking you to turn lights on/off/auto, use "lights.set_state".
- If the user is asking whether the lights are on, off, or what state they are in (e.g. "are the lights on?", "light status", "are my lights on?"), use "lights.get_state".
- If the user asks to run Plex sync, sync Plex, run the Plex sync app, start Plex sync, or similar, use "plex_sync.run".
- If the user is asking about Gmail, email, inbox, or messages, OR is clearly asking
  about things that would usually live in email (for example: literary magazine
  submissions, acceptances/rejections, stories being accepted, submission platforms
  like Submittable or Moksha), use "gmail.search".
  - For gmail.search, set params.query to a SHORT search string that expresses what
    to find. Interpret the user's message: e.g. "last email about literary mag
    submission being accepted" -> "submission accepted literary magazine"; "how many
    unread in Promotions" -> "promotions" or "unread". Use plain words only (no
    subject:, from:, or other operators). This query is sent directly to Gmail.
  - For result_type "list" when the user is asking to find or look through emails
    (e.g. "what's the last email about X", "any acceptance from a literary magazine"),
    also set params.broad_search_terms to an array of 2–8 terms that might appear in
    relevant emails. Include wording that appears in acceptance emails, not just
    rejections: e.g. "feature", "publish", "edits" (as in "work with me on edits"),
    "accepted", "acceptance", "submission", "literary", "magazine". We
    search with OR and then use the model to pick which emails actually match.
  - If the user clearly refers to a specific Gmail category such as "Updates",
    "Promotions", "Primary", "Social", or "Forums", set params.category to the
    corresponding lowercase name (e.g. "Updates" -> "updates").
  - Choose "unread" when they ask about unread, new, or recent mail; otherwise "all".
  - Choose "count" when they ask "how many", "number of", or clearly request a count.
  - Otherwise choose "list".
- If the message is purely conversational or does not match any tool, use "none".

You may use the previous user message and previous action to interpret short
follow-ups like "again", "same thing", or "check that for me". In those cases,
it is usually correct to repeat the same kind of action with similar parameters.

Respond with JSON ONLY, no explanation, in this exact schema:
{{
  "action": "<one of: none, lights.set_state, lights.get_state, plex_sync.run, gmail.search>",
  "params": {{}}
}}

User message:
\"\"\"
{user_text}
\"\"\"

Previous user message:
\"\"\"
{prev_msg}
\"\"\"

Previous routed action (JSON or null):
{prev_route_json}
"""
        try:
            response = ask_lmstudio(routing_prompt)
            raw = response["output"][0]["content"].strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 3:
                    raw = parts[1].strip()
            route = json.loads(raw)
            if not isinstance(route, dict):
                raise ValueError("Routing output was not a JSON object")
            action = str(route.get("action", "none")).strip() or "none"
            params = route.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            cleaned = {"action": action, "params": params}
            if action not in ("none", "lights.set_state", "lights.get_state", "plex_sync.run", "gmail.search"):
                self.log(f"Router returned unknown action {action!r}; using heuristic router.")
                cleaned = self._heuristic_route(user_text)
            self.log(f"Router decided action: {cleaned}")
            return cleaned
        except Exception as e:
            self.log(f"Routing failed; using heuristic router instead: {e}")
            cleaned = self._heuristic_route(user_text)
            self.log(f"Heuristic router decided action: {cleaned}")
            return cleaned

    def _heuristic_route(self, user_text: str) -> dict:
        t = user_text.lower()
        if self.last_route is not None and any(
            phrase in t for phrase in ("again", "same thing", "do that", "check again")
        ):
            return self.last_route
        if "plex" in t and "sync" in t:
            return {"action": "plex_sync.run", "params": {}}
        if "light" in t or "lights" in t:
            if any(phrase in t for phrase in ("are the", "are my", "is the", "status", "state", "check")):
                return {"action": "lights.get_state", "params": {}}
            state = None
            if "auto" in t or "automatic" in t:
                state = "auto"
            elif "on" in t and "off" not in t:
                state = "on"
            elif "off" in t and "on" not in t:
                state = "off"
            if state is not None:
                return {"action": "lights.set_state", "params": {"state": state}}
        if any(w in t for w in ("gmail", "email", "inbox", "mail", "submittable", "moksha")):
            scope = "unread" if any(w in t for w in ("unread", "new", "recent")) else "all"
            result_type = "count" if any(w in t for w in ("how many", "number of", "count")) else "list"
            category = None
            if "updates" in t:
                category = "updates"
            elif "promotions" in t:
                category = "promotions"
            elif "primary" in t:
                category = "primary"
            elif "social" in t:
                category = "social"
            elif "forums" in t:
                category = "forums"
            params = {"query": user_text, "scope": scope, "result_type": result_type}
            if category is not None:
                params["category"] = category
            return {"action": "gmail.search", "params": params}
        return {"action": "none", "params": {}}

    def _call_model(self, prompt: str, light_action: str | None = None, extra_note: str = "") -> str:
        """Build prompt, call LM Studio, return assistant reply text."""
        system_preamble = (
            "You are Galadrial, an AI assistant embedded in a desktop GUI.\n\n"
            "- The app can control my Govee lights via an API. When you see a "
            "system note telling you that the lights were set to a state, you may "
            "speak as if that action has already been performed.\n"
            "- The app may also display separate System messages with the results "
            "of tools (for example, Gmail searches, Plex sync). Those System messages are the "
            "source of truth about my real data.\n\n"
            "CRITICAL SAFETY RULES:\n"
            "- Do NOT invent or guess specific facts about my personal data, such "
            "as emails, literary magazine acceptances, bank balances, calendar "
            "events, or file contents.\n"
            "- If you are not given an explicit System note or tool result that "
            "contains those facts, you must say that you don't know instead of "
            "making something up.\n"
            "- You may still answer general questions with your own knowledge, but "
            "never fabricate concrete details about my life or accounts.\n\n"
            "- The user has a D&D campaign folder (notes and maps). You do not have access to it "
            "from this chat. If they ask, say that the D&D improv feature does: when they open "
            "http://localhost:8000/dnd in a browser (with the API server running), they can "
            "record or paste conversation and click Get suggestion to get dialogue that uses "
            "that folder.\n\n"
        )
        action_note = ""
        if light_action in ("on", "off", "auto"):
            action_note = f"System note: The app has just set the lights to '{light_action}'.\n\n"
        if extra_note:
            action_note = action_note + extra_note
        full_prompt = f"{system_preamble}{action_note}User: {prompt}"
        try:
            response = ask_lmstudio(full_prompt)
            return (response.get("output") or [{}])[0].get("content", "").strip() or "No response."
        except Exception as e:
            self.log(f"Model error: {e}")
            return f"Error calling model: {e}"

    def _interpret_email_list(self, user_question: str, messages: list[dict]) -> str | None:
        """Ask the model which emails match; return reply or None."""
        max_for_interpretation = 45
        if len(messages) > max_for_interpretation:
            messages = messages[:max_for_interpretation]
        if not messages:
            try:
                response = ask_lmstudio(
                    "The user asked about their email:\n\n"
                    f'"{user_question}"\n\n'
                    "We searched their Gmail and found no messages matching the search. "
                    "Reply in one short sentence that no matching emails were found."
                )
                return (response.get("output") or [{}])[0].get("content", "").strip() or None
            except Exception as e:
                self.log(f"Interpretation (no results) failed: {e}")
                return None
        lines = []
        for i, m in enumerate(messages, 1):
            from_ = m.get("from", "")
            subj = m.get("subject", "")
            date = m.get("date", "")
            snippet = m.get("snippet", "")
            parts = [f"{i}. From: {from_}", f"Subject: {subj}"]
            if date:
                parts.append(f"Date: {date}")
            line = " | ".join(parts)
            if snippet:
                line += "\n   Snippet: " + snippet
            lines.append(line)
        email_list_text = "\n".join(lines)
        prompt = (
            "The user asked about their email:\n\n"
            f'"{user_question}"\n\n'
            "Here are emails from their inbox (newest first), with a short body snippet for each:\n\n"
            f"{email_list_text}\n\n"
            "Which of these emails match what they're looking for? Use the snippet to recognize "
            "acceptances: e.g. 'would love to feature', 'we'd like to publish', 'accept your story', "
            "'contract', 'feature it', 'work with me on edits', 'I can publish it', 'publish it in the [issue]', "
            "'earliest available opening', 'would you be willing to work with me'—even if the subject doesn't say 'accepted'. "
            "If one or more match, say which one(s) and give the most relevant detail. "
            "If none match, say so briefly. Reply in 1–4 concise sentences; do not invent any emails not in the list."
        )
        try:
            response = ask_lmstudio(prompt)
            text = (response.get("output") or [{}])[0].get("content", "").strip()
            return text or None
        except Exception as e:
            self.log(f"Interpretation failed: {e}")
            return None

    def _search_gmail_sync(
        self,
        user_question: str,
        query: str,
        scope: str,
        result_type: str,
        category: str | None,
        broad_search_terms: list[str] | None,
    ) -> str:
        """Run Gmail search + optional interpretation; return summary string. Then caller uses it for _call_model."""
        search_query = query
        max_results = 20
        if result_type == "list" and broad_search_terms:
            narrow_terms = [t for t in broad_search_terms if t.lower() not in _GENERIC_BROAD_TERMS]
            if not narrow_terms:
                narrow_terms = list(broad_search_terms)
            search_query = " OR ".join(narrow_terms)
            max_results = 80
            self.log(f"Searching Gmail with broad terms (OR): {search_query!r}")
        self.log(f"Searching Gmail (scope={scope}, result={result_type}) for: {search_query!r}")
        count_only = result_type == "count"
        try:
            messages = search_gmail(
                query=search_query,
                scope=scope,
                max_results=max_results,
                category=category,
                count_only=count_only,
            )
        except GmailClientError as e:
            self.log(f"Gmail error: {e}")
            return self._call_model(
                user_question,
                None,
                extra_note="System note: The app tried to search Gmail but failed. Tell the user briefly that the search failed and they can try again.\n\n",
            )
        total = len(messages)
        thread_count = None
        if count_only and messages and isinstance(messages[0], dict):
            m0 = messages[0]
            if "_count" in m0:
                total = int(m0["_count"])
            if "_thread_count" in m0:
                thread_count = int(m0["_thread_count"])
        if total == 0 and (thread_count is None or thread_count == 0):
            if result_type == "count" and scope == "unread":
                summary = "You have 0 unread Gmail messages."
            elif result_type == "count":
                summary = "You have 0 Gmail messages matching that query."
            else:
                summary = (
                    self._interpret_email_list(user_question.strip(), [])
                    if user_question.strip()
                    else "No matching Gmail messages found."
                )
                summary = summary or "No matching Gmail messages found."
        else:
            if result_type == "count":
                if thread_count is not None and category:
                    summary = f"You have {thread_count} unread conversation(s) in {category} ({total:,} message(s))."
                elif scope == "unread":
                    summary = f"You have {total:,} unread Gmail message(s)."
                else:
                    summary = f"You have {total:,} Gmail message(s) matching that query."
            else:
                if user_question.strip():
                    interpretation = self._interpret_email_list(user_question.strip(), messages)
                    if interpretation:
                        summary = interpretation
                    else:
                        summary = (
                            f"I found {total} email(s) matching your search but couldn't "
                            "interpret which one answers your question. Try asking again."
                        )
                else:
                    lines = [f"- {m.get('from', '')} — {m.get('subject', '')}" for m in messages]
                    summary = f"Found {total} matching Gmail message(s):\n" + "\n".join(lines)
        self.log(summary)
        return self._call_model(
            user_question,
            None,
            extra_note="The app ran a Gmail search. Use this result to answer the user in one concise message. Do not repeat the raw list; summarize or answer their question.\n\nResult:\n" + summary + "\n\n",
        )

    def _run_plex_sync_background(self) -> None:
        try:
            if not os.path.isdir(PLEX_SYNC_DIR) or not os.path.isfile(PLEX_SYNC_MAIN):
                self.log(f"Plex sync path not found: {PLEX_SYNC_DIR!r}")
                return
            python_exe = PLEX_SYNC_PY if os.path.isfile(PLEX_SYNC_PY) else "python"
            subprocess.run(
                [python_exe, PLEX_SYNC_MAIN],
                cwd=PLEX_SYNC_DIR,
                capture_output=True,
                text=True,
            )
            # Optionally log result; for API we don't push back to client
            self.log("Plex sync finished.")
        except Exception as e:
            self.log(f"Plex sync error: {e}")

    def handle_message(self, user_text: str) -> str:
        """
        Route the message, run any tool, then call the model once. Return the assistant reply.
        """
        route = self.route(user_text)
        action = (route.get("action") or "none").lower()
        params = route.get("params") or {}
        self.last_route = route
        self.last_user_message = user_text

        if action == "lights.set_state":
            state = str(params.get("state", "")).lower()
            if state in ("on", "off", "auto"):
                self.log(f"Turning lights {state}.")
                if state == "auto":
                    set_lights_auto()
                else:
                    toggle_all_lights("on" if state == "on" else "off")
                return self._call_model(user_text, state)
            return self._call_model(user_text, None)

        if action == "lights.get_state":
            try:
                result = get_lights_state()
                light_state = result.get("state", "unknown")
                self.log(f"The lights are {light_state}.")
                return self._call_model(
                    user_text,
                    None,
                    extra_note=f"System note: The app just checked the lights; they are {light_state}.\n\n",
                )
            except LightsClientError as e:
                self.log(f"Lights state check failed: {e}")
                return self._call_model(
                    user_text,
                    None,
                    extra_note="System note: The app tried to check the lights but the request failed. Do NOT guess; tell the user the check failed and they can try again.\n\n",
                )

        if action == "gmail.search":
            raw_query = str(params.get("query") or user_text).strip()
            query = " ".join(w for w in raw_query.split() if ":" not in w) or raw_query
            scope = str(params.get("scope") or "unread").lower()
            text_lower = user_text.lower()
            if "unread" in text_lower:
                scope = "unread"
            elif any(w in text_lower for w in (" last ", " latest", "most recent")):
                scope = "all"
            if scope not in ("unread", "all"):
                scope = "unread"
            result_type = str(params.get("result_type") or "list").lower()
            if result_type not in ("count", "list"):
                result_type = "list"
            category = params.get("category")
            if isinstance(category, str):
                category = category.lower()
                if category not in ("updates", "primary", "promotions", "social", "forums"):
                    category = None
            else:
                category = None
            broad_terms = params.get("broad_search_terms")
            if isinstance(broad_terms, list) and result_type == "list":
                broad_terms = [str(t).strip() for t in broad_terms if t and ":" not in str(t)][:8]
            else:
                broad_terms = None
            return self._search_gmail_sync(user_text, query, scope, result_type, category, broad_terms)

        if action == "plex_sync.run":
            self.log("Plex sync started in the background.")
            threading.Thread(target=self._run_plex_sync_background, daemon=True).start()
            return self._call_model(
                user_text,
                None,
                extra_note="System note: The app has just started the Plex sync in the background and will notify when it finishes.\n\n",
            )

        return self._call_model(user_text, None)


def handle_message(user_text: str, log_fn: Optional[Callable[[str], None]] = None) -> str:
    """
    One-shot: create engine, handle message, return reply.
    Use this from FastAPI or scripts. For multi-turn with follow-ups, use AssistantEngine() and call handle_message on it.
    """
    engine = AssistantEngine(log_fn=log_fn)
    return engine.handle_message(user_text)


if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Are the lights on?"
    print("User:", msg)
    reply = handle_message(msg, log_fn=print)
    print("Reply:", reply)
