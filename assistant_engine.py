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
    set_lights_style,
)
from nanoleaf import nanoleaf
from gmail_client import search_gmail, GmailClientError
from pomodoro import get_timer as get_pomodoro_timer

# Plex sync (same as chat_gui; override with PLEX_SYNC_DIR env if needed)
PLEX_SYNC_DIR = os.environ.get("PLEX_SYNC_DIR", r"H:\Coding\Python Projects\plex_sync")
PLEX_SYNC_PY = os.path.join(PLEX_SYNC_DIR, ".venv", "Scripts", "python.exe")
PLEX_SYNC_MAIN = os.path.join(PLEX_SYNC_DIR, "main.py")

# Gmail broad-term filter for list searches
_GENERIC_BROAD_TERMS = {"review", "feature", "contract"}

# Tools the assistant can use. The model sees these and chooses one based on the user's intent.
TOOLS = [
    {
        "name": "none",
        "description": "No tool. Use when the user is just chatting, asking a general question, or when no other tool fits.",
        "params": [],
    },
    {
        "name": "lights.set_state",
        "description": "Turn BOTH Govee and Nanoleaf lights on, off, or set Govee to automatic mode. Use only when the user explicitly says on, off, or auto/automatic—not for moods or scenes.",
        "params": [{"name": "state", "type": "string", "description": "One of: on, off, auto"}],
    },
    {
        "name": "lights.set_scene",
        "description": "Set BOTH Govee and Nanoleaf to a mood or atmosphere. Use when the user asks to change how the lights feel: romantic, peaceful, cozy, somber, pulse, gentle, brighter, faster, or any vibe/intent (e.g. 'add a gentle pulse', 'make them pulse faster and brighter'). We set color and brightness to match; there is no real pulse effect. params.description = the user's full request or mood.",
        "params": [{"name": "description", "type": "string", "description": "The mood or full request (e.g. romantic, somber, gentle pulse, make them brighter)"}],
    },
    {
        "name": "lights.get_state",
        "description": "Check whether Govee lights are currently on or off. Use when the user asks about light status or whether the lights are on.",
        "params": [],
    },
    {
        "name": "plex_sync.run",
        "description": "Run the Plex sync app (syncs media to a server) in the background. Use when the user asks to run Plex sync, sync Plex, or start Plex sync.",
        "params": [],
    },
    {
        "name": "gmail.search",
        "description": "Search the user's Gmail inbox. Use when they ask about email, inbox, messages, or things that would be in email (e.g. literary magazine submissions, acceptances).",
        "params": [
            {"name": "query", "type": "string", "description": "Short search terms (plain words, no operators)"},
            {"name": "scope", "type": "string", "description": "unread or all"},
            {"name": "result_type", "type": "string", "description": "count or list"},
            {"name": "category", "type": "string", "description": "Optional: updates, primary, promotions, social, forums"},
            {"name": "broad_search_terms", "type": "array", "description": "Optional: list of terms for list searches to widen results"},
        ],
    },
    {
        "name": "nanoleaf.set_scene",
        "description": "Set ONLY the Nanoleaf panels to a predefined SCENE from the list (e.g. Romantic, Northern Lights, Inner Peace). Use when the user asks for a mood that fits a scene name. Do NOT use for: pulse, 'add a pulse', breathing, 'custom', 'without using scenes', 'make up your own'—use nanoleaf.custom for those.",
        "params": [{"name": "description", "type": "string", "description": "The mood that should match a scene name (e.g. romantic, peaceful)"}],
    },
    {
        "name": "nanoleaf.custom",
        "description": "Set BOTH Govee and Nanoleaf from a custom/pulse-style request. Use when the user asks for a pulse, 'add a pulse', 'strong pulse', animation, rhythm, or says 'without using the scenes list' / 'make up your own settings' / 'custom'. Nanoleaf can show real animations (we pick a pulse/animation scene if one fits); Govee can only do a single static color and brightness (no animation). Do NOT use for a simple mood that matches a scene—use nanoleaf.set_scene or lights.set_scene instead.",
        "params": [{"name": "description", "type": "string", "description": "The request (e.g. strong pulse, gentle pulse, custom somber, make up your own)"}],
    },
    {
        "name": "nanoleaf.create_animation",
        "description": "CREATE AND APPLY a new animation on the NANOLEAF panels only (e.g. a flowing cycle of colors). The app will run this on the Nanoleaf—do NOT reply with manual instructions for the user. Use when the user says: create a new animation, create an animation for the lights, new animation for lighting, make a flowing effect, cycle through colors, or design an animation. Pick 2–6 colors (hex) and optional speed; use animation_type 'flow', params.colors (e.g. [\"#FF0000\", \"#00FF00\"]), params.speed 0.5–5. Never respond with text telling the user to set Govee in an app—use this tool to apply on Nanoleaf.",
        "params": [
            {"name": "animation_type", "type": "string", "description": "One of: flow"},
            {"name": "colors", "type": "array", "description": "List of hex color strings, e.g. [\"#FF0000\", \"#00FF00\", \"#0000FF\"]. Need at least 2 for flow."},
            {"name": "speed", "type": "number", "description": "Optional. Transition speed 0.5–5 (seconds). Default 1."},
        ],
    },
    {
        "name": "pomodoro.start",
        "description": "Start a pomodoro focus session (25 min work, then automatic break). Lights shift to a cool, bright focus setting. Use when the user says: start a focus session, start pomodoro, let's focus, time to work, begin a pomodoro, focus mode.",
        "params": [],
    },
    {
        "name": "pomodoro.stop",
        "description": "Stop the current pomodoro timer and return to idle. Use when the user says: stop the timer, cancel pomodoro, end the session, I'm done focusing.",
        "params": [],
    },
    {
        "name": "pomodoro.skip",
        "description": "Skip the current pomodoro phase (jump from focus to break or break to focus). Use when the user says: skip, next phase, skip the break, skip to break, move on.",
        "params": [],
    },
    {
        "name": "pomodoro.status",
        "description": "Check the current pomodoro timer status: what phase, time remaining, how many completed today. Use when the user asks: how many pomodoros today, what's the timer at, am I on a break, pomodoro stats, how much time left, timer status.",
        "params": [],
    },
]

VALID_ACTIONS = {t["name"] for t in TOOLS}


def _format_tools_for_prompt() -> str:
    """Format TOOLS into a string for the LLM prompt."""
    lines = []
    for t in TOOLS:
        params_str = ""
        if t["params"]:
            params_str = " Parameters: " + ", ".join(
                f'{p["name"]} ({p["type"]})' + (f": {p['description']}" if p.get("description") else "")
                for p in t["params"]
            )
        lines.append(f'- {t["name"]}: {t["description"]}{params_str}')
    return "\n".join(lines)


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

    def _route_speed_adjust(self, user_text: str) -> dict | None:
        """If the user is asking to change animation speed and last action was nanoleaf.create_animation, return a route that re-applies the same animation with new speed. Otherwise return None."""
        if self.last_route is None or self.last_route.get("action") != "nanoleaf.create_animation":
            return None
        prev_params = self.last_route.get("params") or {}
        colors = prev_params.get("colors")
        if not isinstance(colors, list) or len(colors) < 2:
            return None
        t = user_text.lower().strip()
        try:
            current = float(prev_params.get("speed", 1.0))
        except (TypeError, ValueError):
            current = 1.0
        current = max(0.5, min(5.0, current))
        # Faster: lower numeric speed = faster transition in our API
        if any(
            phrase in t
            for phrase in (
                "super fast",
                "really fast",
                "maximum speed",
                "max speed",
                "faster",
                "speed it up",
                "speed up",
                "a little faster",
                "make it faster",
                "flow faster",
            )
        ):
            if "super" in t or "really fast" in t or "max" in t:
                new_speed = 0.5
            elif "a little" in t:
                new_speed = max(0.5, round(current * 0.75, 1))
            else:
                new_speed = max(0.5, round(current * 0.55, 1))
            return {
                "action": "nanoleaf.create_animation",
                "params": {
                    "animation_type": "flow",
                    "colors": colors,
                    "speed": new_speed,
                },
            }
        # Slower: higher numeric speed = slower transition
        if any(
            phrase in t
            for phrase in (
                "slower",
                "slow it down",
                "slow down",
                "a little slower",
                "make it slower",
                "more slowly",
                "gentler",
            )
        ):
            if "a little" in t:
                new_speed = min(5.0, round(current * 1.25, 1))
            else:
                new_speed = min(5.0, round(current * 1.6, 1))
            return {
                "action": "nanoleaf.create_animation",
                "params": {
                    "animation_type": "flow",
                    "colors": colors,
                    "speed": new_speed,
                },
            }
        return None

    def _route_animation_color_change(self, user_text: str) -> dict | None:
        """If the user is asking to change the lights to a single color (e.g. 'make the lights yellow'), return nanoleaf.create_animation with that color so we get a yellow flow on Nanoleaf instead of a scene. Reuse previous speed if last action was create_animation, else default 1.0."""
        prev_params = (self.last_route or {}).get("params") or {} if self.last_route and self.last_route.get("action") == "nanoleaf.create_animation" else {}
        try:
            speed = float(prev_params.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.5, min(5.0, speed))
        t = user_text.lower().strip()
        # "make the lights yellow", "make them yellow", "make it blue", "change to red", "turn them green"
        color_words = {
            "yellow": ["#FFFF00", "#FFD700"],
            "gold": ["#FFD700", "#FFA500"],
            "blue": ["#0066FF", "#3399FF"],
            "red": ["#FF0000", "#CC0000"],
            "green": ["#00FF00", "#00CC00"],
            "orange": ["#FF8C00", "#FFA500"],
            "purple": ["#800080", "#9932CC"],
            "violet": ["#EE82EE", "#9932CC"],
            "pink": ["#FF69B4", "#FFB6C1"],
            "white": ["#FFFFFF", "#E8E8E8"],
            "cyan": ["#00FFFF", "#40E0D0"],
            "teal": ["#008080", "#20B2AA"],
            "indigo": ["#4B0082", "#6A0DAD"],
            "amber": ["#FFBF00", "#FF8C00"],
        }
        for word, hexes in color_words.items():
            if word in t and any(
                phrase in t
                for phrase in (
                    "make them", "make it", "make the lights", "make the light",
                    "change to", "turn them", "turn it", "set them", "set it",
                    "want ", "go ", "switch to", "make the", "them ", "lights ",
                )
            ):
                return {
                    "action": "nanoleaf.create_animation",
                    "params": {
                        "animation_type": "flow",
                        "colors": hexes,
                        "speed": speed,
                    },
                }
        # Also catch "make them <color>" when color is at end
        for word, hexes in color_words.items():
            if t.endswith(word) or t.endswith(word + ".") or f" {word}" in t or f" {word}." in t:
                if len(t) < 50 and not t.startswith("set the nanoleaf") and "scene" not in t:
                    return {
                        "action": "nanoleaf.create_animation",
                        "params": {
                            "animation_type": "flow",
                            "colors": hexes,
                            "speed": speed,
                        },
                    }
        return None

    def route(self, user_text: str) -> dict:
        """Return {action, params} by asking the model to choose a tool from TOOLS based on the user's intent. No keyword routing."""
        prev_msg = self.last_user_message or ""
        prev_action = json.dumps(self.last_route) if self.last_route is not None else "null"
        # If user wants to change animation speed and we have a previous create_animation, reuse it with new speed so we actually apply
        speed_route = self._route_speed_adjust(user_text)
        if speed_route is not None:
            self.log(f"Router decided action (speed adjust): {speed_route}")
            return speed_route
        # If user asks to change the animation color (e.g. "make them yellow") and we're already in a flow, keep the flow and just change colors—don't switch to a scene
        color_route = self._route_animation_color_change(user_text)
        if color_route is not None:
            self.log(f"Router decided action (animation color change): {color_route}")
            return color_route
        tools_blob = _format_tools_for_prompt()

        tool_choice_prompt = f"""You are Galadrial, a desktop assistant. You have access to these tools. Interpret the user's message and choose ONE tool (or "none" only if they are clearly just chatting, not asking you to do something).

TOOLS:
{tools_blob}

Interpret intent from the actual request. Examples:
- "set the lights to romantic" / "romantic feel" / "make it cozy" / "add a gentle pulse" / "make them pulse faster and brighter" → lights.set_scene (params.description = their request or mood)
- "set the nanoleaf to romantic" (mood that matches a scene) → nanoleaf.set_scene
- "add a strong pulse to the nanoleaf" / "pulse" / "without using the scenes list" / "make up your own settings" → nanoleaf.custom (custom color/brightness, no scene)
- "create a new animation" / "create a new animation for the lights" / "create an animation for lighting" / "make a flow of red and blue" → nanoleaf.create_animation. You MUST call this tool and supply colors (and optional speed); the app will apply it to the Nanoleaf. Do NOT choose "none" or reply with manual instructions for Govee or any app.
- "make it faster" / "speed it up" / "super fast" / "slower" / "slow it down" (when previous action was nanoleaf.create_animation) → nanoleaf.create_animation again with the SAME colors from the previous action and a NEW speed (faster = lower speed number e.g. 0.5, slower = higher e.g. 2). You must call the tool, not just describe the change.
- "make the lights yellow" / "make them yellow" / "make it blue" / "change to red" → nanoleaf.create_animation with that color as params.colors (e.g. ["#FFFF00", "#FFD700"]). Do NOT use lights.set_scene or nanoleaf.set_scene for a single-color request—use nanoleaf.create_animation so Nanoleaf shows a flowing animation in that color.
- "turn lights on" / "lights off" → lights.set_state with state "on" or "off"
- "are the lights on?" → lights.get_state
- "run Plex sync" → plex_sync.run
- "check my email for X" → gmail.search with query/scope/result_type
- "start a focus session" / "pomodoro" / "time to work" / "focus mode" → pomodoro.start
- "stop the timer" / "cancel the pomodoro" / "I'm done" → pomodoro.stop
- "skip" / "skip the break" / "next phase" → pomodoro.skip
- "how many pomodoros today?" / "timer status" / "how much time left" → pomodoro.status

If the user said "again" or "same thing", repeat the previous action. Do not choose "none" when the user is asking you to change the lights or set a mood—use lights.set_scene or nanoleaf.set_scene as appropriate. If they ask to "create a new animation" or "create an animation for the lights", always use nanoleaf.create_animation with colors (and optional speed)—never respond with a list of settings for them to enter in an app.

Previous user message: "{prev_msg}"
Previous action: {prev_action}

User message: "{user_text}"

Respond with JSON only:
{{"action": "<tool name from the list above>", "params": {{...}}}}
"""
        try:
            response = ask_lmstudio(tool_choice_prompt)
            raw = response["output"][0]["content"].strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1].strip()
            route = json.loads(raw)
            if not isinstance(route, dict):
                raise ValueError("Response was not a JSON object")
            action = str(route.get("action", "none")).strip() or "none"
            params = route.get("params") if isinstance(route.get("params"), dict) else {}
            cleaned = {"action": action, "params": params}
            if action not in VALID_ACTIONS:
                self.log(f"Model returned unknown action {action!r}; using fallback.")
                cleaned = self._heuristic_route(user_text)
            self.log(f"Router decided action: {cleaned}")
            return cleaned
        except Exception as e:
            self.log(f"Tool choice failed ({e}); using fallback.")
            cleaned = self._heuristic_route(user_text)
            self.log(f"Router decided action: {cleaned}")
            return cleaned

    def _heuristic_route(self, user_text: str) -> dict:
        """Used only when the model fails or returns an unknown action. Minimal: repeat last action or none."""
        t = user_text.lower()
        if self.last_route is not None and any(
            phrase in t for phrase in ("again", "same thing", "do that", "check again")
        ):
            return self.last_route
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
            "- The app has a pomodoro focus timer that integrates with the lights. "
            "When a pomodoro starts, lights shift to a cool focus setting; on breaks, "
            "they shift to warm relaxation. Give brief, contextual encouragement: "
            "motivational at the start, congratulatory on completion, gentle during breaks.\n"
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

    def _pick_govee_style(self, description: str) -> dict | None:
        """Ask LLM for color_hex, color_temp_k, brightness that fit the mood. Returns dict or None."""
        prompt = (
            f'The user wants their room lights to feel "{description}".\n\n'
            "Suggest settings for a smart bulb: color (hex #RRGGBB), color temperature in Kelvin (2200–3000 warm, 4000–6500 cool), and brightness 0–100. "
            "You MUST always include brightness. If they say dimmer, dim, or dark use brightness 25–45; if they say brighter or bright use 75–100; otherwise 50–70. "
            "Reply with JSON only, no explanation:\n"
            '{"color_hex": "#RRGGBB", "color_temp_k": 2700, "brightness": 70}'
        )
        try:
            response = ask_lmstudio(prompt)
            raw = (response.get("output") or [{}])[0].get("content", "").strip()
            if not raw:
                return None
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1].strip()
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                return None
            color_hex = obj.get("color_hex")
            color_temp_k = obj.get("color_temp_k")
            brightness = obj.get("brightness")
            if color_hex is None and color_temp_k is None and brightness is None:
                return None
            out = {}
            if color_hex is not None:
                s = str(color_hex).strip()
                if s.startswith("#") and len(s) == 7:
                    out["color_hex"] = s
            if color_temp_k is not None:
                try:
                    out["color_temp_k"] = max(2200, min(6500, int(color_temp_k)))
                except (TypeError, ValueError):
                    pass
            if brightness is not None:
                try:
                    out["brightness"] = max(0, min(100, int(brightness)))
                except (TypeError, ValueError):
                    pass
            # If we have any style but LLM omitted brightness, infer from description so dimmer/brighter are applied
            if out and out.get("brightness") is None:
                d = description.lower()
                if any(w in d for w in ("dim", "dark", "darker", "low")):
                    out["brightness"] = 35
                elif any(w in d for w in ("bright", "brighter", "lighter", "high")):
                    out["brightness"] = 85
                else:
                    out["brightness"] = 65
            return out if out else None
        except Exception as e:
            self.log(f"Govee style pick failed: {e}")
            return None

    def _pick_nanoleaf_style(self, description: str) -> dict | None:
        """Ask LLM for color_hex and brightness for Nanoleaf panels to match the mood. Returns dict or None."""
        prompt = (
            f'The user wants their Nanoleaf panels to feel "{description}".\n\n'
            "Suggest a single color (hex #RRGGBB) and brightness (0–100). You MUST always include brightness. "
            "If they say dimmer, dim, or dark use brightness 25–45; if they say brighter or bright use 75–100; otherwise 50–70. "
            "Reply with JSON only, no explanation:\n"
            '{"color_hex": "#RRGGBB", "brightness": 60}'
        )
        try:
            response = ask_lmstudio(prompt)
            raw = (response.get("output") or [{}])[0].get("content", "").strip()
            if not raw:
                return None
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1].strip()
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                return None
            out = {}
            color_hex = obj.get("color_hex")
            if color_hex is not None:
                s = str(color_hex).strip()
                if s.startswith("#") and len(s) == 7:
                    out["color_hex"] = s
            brightness = obj.get("brightness")
            if brightness is not None:
                try:
                    out["brightness"] = max(0, min(100, int(brightness)))
                except (TypeError, ValueError):
                    pass
            # If we have color but LLM omitted brightness, infer from description
            if out and out.get("brightness") is None:
                d = description.lower()
                if any(w in d for w in ("dim", "dark", "darker", "low")):
                    out["brightness"] = 35
                elif any(w in d for w in ("bright", "brighter", "lighter", "high")):
                    out["brightness"] = 85
                else:
                    out["brightness"] = 60
            return out if out else None
        except Exception as e:
            self.log(f"Nanoleaf style pick failed: {e}")
            return None

    def _pick_nanoleaf_scene(self, description: str, scene_list: list[str]) -> str | None:
        """Use the LLM to pick the scene that best matches the user's mood or intent (e.g. 'sexy' -> Romantic)."""
        if not scene_list or not description.strip():
            return None
        scene_list_str = ", ".join(scene_list)
        prompt = (
            f"Available Nanoleaf scene names (pick exactly one):\n{scene_list_str}\n\n"
            f"The user said they want: \"{description.strip()}\"\n\n"
            "They may use words that are not exact scene names (e.g. 'sexy', 'chill', 'cozy', 'intimate'). "
            "Choose the scene that best fits the mood or feeling they are describing. "
            "For example, 'sexy' or 'intimate' are similar to Romantic; 'calm' might match Inner Peace or Snowfall. "
            "Reply with ONLY the single scene name from the list above—no explanation, no quotes, just the name."
        )
        try:
            response = ask_lmstudio(prompt)
            raw = (response.get("output") or [{}])[0].get("content", "").strip()
            if not raw:
                return None
            # Take first line and strip quotes
            first_line = raw.split("\n")[0].strip().strip('"\'')
            first_line_lower = first_line.lower()
            for name in scene_list:
                if name.lower() == first_line_lower:
                    return name
            # Partial match (e.g. "Inner Peace" in "Inner Peace")
            for name in scene_list:
                if name.lower() in first_line_lower or first_line_lower in name.lower():
                    return name
            return None
        except Exception as e:
            self.log(f"Nanoleaf scene pick failed: {e}")
            return None

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
                    # Nanoleaf has no auto mode; leave as-is or could turn off—leaving as-is
                else:
                    toggle_all_lights("on" if state == "on" else "off")
                    try:
                        if state == "on":
                            nanoleaf.turn_on()
                        else:
                            nanoleaf.turn_off()
                    except Exception as e:
                        self.log(f"Nanoleaf set_state {state} failed: {e}")
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

        if action == "lights.set_scene":
            description = str(params.get("description") or user_text).strip()
            govee_ok = False
            govee_note = ""
            style = self._pick_govee_style(description)
            if style:
                try:
                    set_lights_style(state="on", **style)
                    govee_ok = True
                    parts = ["Govee lights set to match the mood"]
                    if style.get("color_hex"):
                        parts.append(f"(color {style['color_hex']})")
                    if style.get("brightness") is not None:
                        parts.append(f"brightness {style['brightness']}%")
                    govee_note = " ".join(parts) + "."
                    self.log(govee_note)
                except LightsClientError as e:
                    self.log(f"Govee set_style failed: {e}; falling back to on only.")
                    try:
                        toggle_all_lights("on")
                        govee_ok = True
                        govee_note = "Govee lights turned on (style endpoint unavailable)."
                    except LightsClientError:
                        pass
            else:
                try:
                    toggle_all_lights("on")
                    govee_ok = True
                    govee_note = "Govee lights turned on."
                    self.log("Govee lights turned on.")
                except LightsClientError as e:
                    self.log(f"Govee on failed: {e}")
            scenes = nanoleaf.get_scene_list()
            nanoleaf_note = ""
            chosen = self._pick_nanoleaf_scene(description, scenes) if scenes else None
            if chosen:
                try:
                    nanoleaf.turn_on()
                    nanoleaf.set_effect(chosen)
                    self.log(f"Nanoleaf scene set to {chosen!r}.")
                    nanoleaf_note = f"Nanoleaf panels set to scene \"{chosen}\"."
                    # If user asked for dimmer/brighter, apply brightness on top of the scene
                    desc_lower = description.lower()
                    if any(w in desc_lower for w in ("dim", "bright", "brightness", "darker", "lighter", "low", "high")):
                        nl_style = self._pick_nanoleaf_style(description)
                        if nl_style and nl_style.get("brightness") is not None:
                            try:
                                nanoleaf.set_brightness(nl_style["brightness"])
                                nanoleaf_note += f" Brightness set to {nl_style['brightness']}%."
                            except Exception:
                                pass
                except Exception as e:
                    self.log(f"Nanoleaf set_effect failed: {e}")
                    nanoleaf_note = f"Failed to set Nanoleaf to \"{chosen}\" (panels may be unreachable)."
            else:
                # No matching scene: let the LLM pick a color and brightness that fit the mood
                style = self._pick_nanoleaf_style(description)
                if style:
                    try:
                        nanoleaf.turn_on()
                        if style.get("color_hex"):
                            h = style["color_hex"]
                            r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
                            nanoleaf.set_color_rgb(r, g, b)
                        if style.get("brightness") is not None:
                            nanoleaf.set_brightness(style["brightness"])
                        self.log("Nanoleaf set to custom color/brightness for mood.")
                        nanoleaf_note = "Nanoleaf panels set to a custom color and brightness to match the mood."
                    except Exception as e:
                        self.log(f"Nanoleaf style failed: {e}")
                        nanoleaf_note = "Could not set Nanoleaf color/brightness (panels may be unreachable)."
                else:
                    nanoleaf_note = "No matching Nanoleaf scene and could not pick a custom color; try a different description."
            parts = []
            if govee_ok and govee_note:
                parts.append(govee_note)
            elif govee_ok:
                parts.append("Govee lights are on.")
            if nanoleaf_note:
                parts.append(nanoleaf_note)
            extra_note = "System note: " + " ".join(parts) + "\n\n"
            return self._call_model(user_text, None, extra_note=extra_note)

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

        if action == "nanoleaf.set_scene":
            description = str(params.get("description") or user_text).strip()
            scenes = nanoleaf.get_scene_list()
            if not scenes:
                return self._call_model(
                    user_text,
                    None,
                    extra_note="System note: The app could not read the Nanoleaf scenes list (scenes.txt). Tell the user to add scene names to nanoleaf/scenes.txt.\n\n",
                )
            chosen = self._pick_nanoleaf_scene(description, scenes)
            if not chosen:
                return self._call_model(
                    user_text,
                    None,
                    extra_note="System note: The app could not pick a matching Nanoleaf scene from the list. Suggest the user try a different description or use 'custom' for a pulse/custom look.\n\n",
                )
            try:
                nanoleaf.turn_on()
                nanoleaf.set_effect(chosen)
                self.log(f"Nanoleaf scene set to {chosen!r}.")
                return self._call_model(
                    user_text,
                    None,
                    extra_note=f"System note: The app has set the Nanoleaf panels to the scene \"{chosen}\".\n\n",
                )
            except Exception as e:
                self.log(f"Nanoleaf set_effect failed: {e}")
                return self._call_model(
                    user_text,
                    None,
                    extra_note=f"System note: The app tried to set the Nanoleaf scene to \"{chosen}\" but the request failed. Tell the user to check the panels are on and reachable.\n\n",
                )

        if action == "nanoleaf.custom":
            description = str(params.get("description") or user_text).strip()
            govee_style = self._pick_govee_style(description)
            scenes = nanoleaf.get_scene_list()
            # Try an animated/pulse scene for Nanoleaf when user asks for pulse, animation, rhythm, etc.
            nanoleaf_scene = None
            if scenes and any(w in description.lower() for w in ("pulse", "animation", "animate", "rhythm", "beat", "moving", "dynamic")):
                nanoleaf_scene = self._pick_nanoleaf_scene(description, scenes)
            nanoleaf_style = self._pick_nanoleaf_style(description) if not nanoleaf_scene else None
            if not govee_style and not nanoleaf_scene and not nanoleaf_style:
                return self._call_model(
                    user_text,
                    None,
                    extra_note="System note: The app could not pick a custom look for the lights. Ask the user to try again or describe the look they want.\n\n",
                )
            parts = []
            if govee_style:
                try:
                    set_lights_style(state="on", **govee_style)
                    self.log("Govee set to custom color/brightness.")
                    parts.append("Govee lights set to a static color and brightness (Govee cannot do pulse or animation).")
                except LightsClientError as e:
                    self.log(f"Govee custom failed: {e}")
                    parts.append("Govee lights could not be updated.")
            if nanoleaf_scene:
                try:
                    nanoleaf.turn_on()
                    nanoleaf.set_effect(nanoleaf_scene)
                    self.log(f"Nanoleaf set to scene {nanoleaf_scene!r} (animated).")
                    parts.append(f"Nanoleaf panels set to the \"{nanoleaf_scene}\" scene (animated/pulse).")
                except Exception as e:
                    self.log(f"Nanoleaf set_effect failed: {e}")
                    parts.append("Nanoleaf panels could not be updated.")
            elif nanoleaf_style:
                try:
                    nanoleaf.turn_on()
                    if nanoleaf_style.get("color_hex"):
                        h = nanoleaf_style["color_hex"]
                        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
                        nanoleaf.set_color_rgb(r, g, b)
                    if nanoleaf_style.get("brightness") is not None:
                        nanoleaf.set_brightness(nanoleaf_style["brightness"])
                    self.log("Nanoleaf set to custom color/brightness.")
                    parts.append("Nanoleaf panels set to a custom color and brightness (static).")
                except Exception as e:
                    self.log(f"Nanoleaf custom failed: {e}")
                    parts.append("Nanoleaf panels could not be updated.")
            note = " ".join(parts)
            return self._call_model(
                user_text,
                None,
                extra_note="System note: " + note + "\n\n",
            )

        if action == "nanoleaf.create_animation":
            anim_type = str(params.get("animation_type") or "flow").strip().lower()
            raw_colors = params.get("colors")
            if isinstance(raw_colors, list) and len(raw_colors) >= 2:
                colors_rgb = []
                for c in raw_colors:
                    s = str(c).strip()
                    if s.startswith("#") and len(s) >= 7:
                        try:
                            r = int(s[1:3], 16)
                            g = int(s[3:5], 16)
                            b = int(s[5:7], 16)
                            colors_rgb.append((r, g, b))
                        except ValueError:
                            pass
                if len(colors_rgb) >= 2:
                    speed = params.get("speed")
                    try:
                        speed = float(speed) if speed is not None else 1.0
                    except (TypeError, ValueError):
                        speed = 1.0
                    speed = max(0.5, min(5.0, round(speed, 1)))
                    try:
                        ok = nanoleaf.create_flow_effect(colors_rgb, speed)
                        if ok:
                            self.log("Nanoleaf create_flow_effect succeeded.")
                            # If user asked to "make the lights" a color, set Govee to that color too
                            first_hex = raw_colors[0] if raw_colors else None
                            if first_hex:
                                try:
                                    set_lights_style(state="on", color_hex=first_hex, brightness=75)
                                    self.log("Govee set to match animation color.")
                                except LightsClientError:
                                    pass
                            return self._call_model(
                                user_text,
                                None,
                                extra_note="System note: A new flowing animation was created on the Nanoleaf panels with the chosen colors and speed."
                                + (" Govee lights were also set to match." if first_hex else "") + "\n\n",
                            )
                    except Exception as e:
                        self.log(f"nanoleaf.create_animation failed: {e}")
            return self._call_model(
                user_text,
                None,
                extra_note="System note: Could not create the animation (need at least 2 valid hex colors in params.colors, e.g. [\"#FF0000\", \"#0000FF\"]).\n\n",
            )

        if action == "plex_sync.run":
            self.log("Plex sync started in the background.")
            threading.Thread(target=self._run_plex_sync_background, daemon=True).start()
            return self._call_model(
                user_text,
                None,
                extra_note="System note: The app has just started the Plex sync in the background and will notify when it finishes.\n\n",
            )

        if action == "pomodoro.start":
            timer = get_pomodoro_timer(log_fn=self.log)
            result = timer.start_focus()
            completed = result.get("completed_today", 0)
            duration = result.get("phase_duration_seconds", 1500) // 60
            return self._call_model(
                user_text,
                None,
                extra_note=(
                    f"System note: A {duration}-minute focus session has started. "
                    f"Lights are set to a cool, bright focus setting. "
                    f"Completed pomodoros today: {completed}. "
                    "Give the user a brief, motivational message to start their focus session.\n\n"
                ),
            )

        if action == "pomodoro.stop":
            timer = get_pomodoro_timer(log_fn=self.log)
            result = timer.stop()
            completed = result.get("completed_today", 0)
            return self._call_model(
                user_text,
                None,
                extra_note=(
                    f"System note: The pomodoro timer has been stopped. "
                    f"Total completed pomodoros today: {completed}. "
                    "Acknowledge the stop and mention their progress.\n\n"
                ),
            )

        if action == "pomodoro.skip":
            timer = get_pomodoro_timer(log_fn=self.log)
            result = timer.skip()
            new_state = result.get("state", "idle")
            remaining = result.get("remaining_seconds", 0)
            return self._call_model(
                user_text,
                None,
                extra_note=(
                    f"System note: Skipped to the next phase. Now in '{new_state}' "
                    f"with {remaining // 60} minutes remaining. "
                    "Briefly tell the user what phase they are now in.\n\n"
                ),
            )

        if action == "pomodoro.status":
            timer = get_pomodoro_timer(log_fn=self.log)
            result = timer.get_status()
            state = result.get("state", "idle")
            completed = result.get("completed_today", 0)
            remaining = result.get("remaining_seconds", 0)
            if state == "idle":
                extra = (
                    f"System note: No pomodoro is running. "
                    f"Completed today: {completed}. "
                    "Tell the user their stats and offer to start a new session.\n\n"
                )
            else:
                extra = (
                    f"System note: Pomodoro is in '{state}' phase. "
                    f"{remaining // 60}m {remaining % 60}s remaining. "
                    f"Completed today: {completed}. "
                    "Summarize the status concisely.\n\n"
                )
            return self._call_model(user_text, None, extra_note=extra)

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
