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
from daily_briefing import (
    gather_briefing_data,
    format_briefing_for_llm,
    get_time_of_day_light_mood,
    get_goodnight_light_mood,
)

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
        "description": "Set Govee lights to a mood/scene. Use for requests like 'set a cozy mood', 'make it romantic', etc.",
        "params": [{"name": "mood", "type": "string", "description": "Scene name: cozy, romantic, productive, chill, gaming, party"}],
    },
    {
        "name": "lights.set_nanoleaf",
        "description": "Set Nanoleaf to a specific hue/saturation/brightness. Use for colour changes, not scenes.",
        "params": [
            {"name": "hue", "type": "number", "description": "0-360 (optional)"},
            {"name": "saturation", "type": "number", "description": "0-100 (optional)"},
            {"name": "brightness", "type": "number", "description": "0-100 (optional)"},
        ],
    },
    {
        "name": "briefing.morning",
        "description": "Morning briefing: set a bright energizing light mood, check email, weather, and calendar. Say good morning!",
        "params": [],
    },
    {
        "name": "briefing.goodnight",
        "description": "Good night routine: dim lights to a warm sleepy glow and say goodnight.",
        "params": [],
    },
    {
        "name": "email.search",
        "description": "Search email by label or keywords. Use for questions like 'Do I have any emails about contracts?' or 'Show me reviews'.",
        "params": [{"name": "query", "type": "string", "description": "Email label or keyword search"}],
    },
    {
        "name": "plex_sync.run",
        "description": "Start a Plex library sync in the background.",
        "params": [],
    },
]

# Routing: each route maps natural language triggers (via LLM) to tool dispatch.
# The model reads these and picks the best tool based on the user's intent.
ROUTES = [
    {
        "tool": "lights.set_state",
        "triggers": ["turn on", "turn off", "turn the lights on", "turn the lights off", "lights on", "lights off"],
    },
    {
        "tool": "lights.set_scene",
        "triggers": ["set a mood", "set a scene", "cozy", "romantic", "productive", "chill", "gaming", "party"],
    },
    {
        "tool": "lights.set_nanoleaf",
        "triggers": ["change color", "set color", "hue", "saturation", "brightness"],
    },
    {
        "tool": "briefing.morning",
        "triggers": ["morning briefing", "good morning", "morning", "wake up", "start my day", "briefing"],
    },
    {
        "tool": "briefing.goodnight",
        "triggers": ["goodnight", "good night", "sleep", "bedtime", "wind down", "sleep time"],
    },
    {
        "tool": "email.search",
        "triggers": ["email", "mail", "check email", "any emails", "new emails", "search email"],
    },
    {
        "tool": "plex_sync.run",
        "triggers": ["sync plex", "plex sync", "update plex", "refresh plex"],
    },
]


class AssistantEngine:
    """
    Stateful assistant: route user message → run tools → call LLM → return reply string.
    Can be reused for multi-turn conversations.
    """

    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda x: None)

    def log(self, msg: str):
        """Utility to print debug logs."""
        self.log_fn(msg)

    def _call_model(self, user_text: str, tool_result: Optional[str] = None, extra_note: str = ""):
        """
        Ask the LLM to generate a reply.
        If tool_result is provided, the model sees both the user's message and the tool output.
        """
        prompt = extra_note
        if tool_result:
            prompt += f"Tool result:\n{tool_result}\n\n"
        prompt += f"User: {user_text}\n\nAssistant:"

        reply = ask_lmstudio(prompt)
        return reply.strip()

    def _choose_tool(self, user_text: str) -> str:
        """
        Ask the LLM to pick a tool based on the user's message.
        Returns the tool name or "none".
        """
        tool_names = [t["name"] for t in TOOLS]
        tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in TOOLS])

        prompt = f"""You are an assistant that routes user messages to the best tool.

Available tools:
{tool_descriptions}

User message: "{user_text}"

Respond with ONLY the tool name (e.g., "lights.set_state", "email.search", "none"). No explanation."""

        response = ask_lmstudio(prompt)
        chosen_tool = response.strip().lower()

        # Validate the tool name
        if chosen_tool not in tool_names:
            return "none"
        return chosen_tool

    def _run_plex_sync_background(self):
        """Run plex sync in a subprocess."""
        try:
            self.log(f"Running Plex sync from {PLEX_SYNC_MAIN}")
            subprocess.run(
                [PLEX_SYNC_PY, PLEX_SYNC_MAIN],
                capture_output=True,
                timeout=300,
            )
            self.log("Plex sync completed.")
        except subprocess.TimeoutExpired:
            self.log("Plex sync timed out.")
        except Exception as e:
            self.log(f"Plex sync error: {e}")

    def handle_message(self, user_text: str) -> str:
        """
        Handle one user message: route to a tool, run it, call LLM, return reply.
        """
        self.log(f"User: {user_text}")

        # Choose a tool
        action = self._choose_tool(user_text)
        self.log(f"Tool chosen: {action}")

        # Route to the tool's handler
        if action == "lights.set_state":
            # Ask the model to extract the state (on, off, auto)
            state_prompt = f"""Extract the desired light state from this message: "{user_text}"
Respond with ONLY one of: on, off, auto. No explanation."""
            state = ask_lmstudio(state_prompt).strip().lower()
            if state not in ["on", "off", "auto"]:
                state = "on"

            try:
                toggle_all_lights(state)
                result = f"Lights turned {state}."
            except LightsClientError as e:
                result = f"Error: {e}"

            self.log(result)
            return self._call_model(user_text, result)

        if action == "lights.set_scene":
            # Ask the model to extract the mood
            mood_prompt = f"""Extract the desired mood/scene from this message: "{user_text}"
Respond with ONLY one of: cozy, romantic, productive, chill, gaming, party. No explanation."""
            mood = ask_lmstudio(mood_prompt).strip().lower()
            if mood not in ["cozy", "romantic", "productive", "chill", "gaming", "party"]:
                mood = "cozy"

            try:
                set_lights_style(mood)
                result = f"Lights set to {mood} mood."
            except LightsClientError as e:
                result = f"Error: {e}"

            self.log(result)
            return self._call_model(user_text, result)

        if action == "lights.set_nanoleaf":
            # Ask the model to extract hue, saturation, brightness
            params_prompt = f"""Extract light parameters from this message: "{user_text}"
Respond with JSON (use null for omitted): {{"hue": 0-360 or null, "saturation": 0-100 or null, "brightness": 0-100 or null}}
Example: {{"hue": 180, "saturation": null, "brightness": 80}}"""
            params_str = ask_lmstudio(params_prompt).strip()

            try:
                params = json.loads(params_str)
            except json.JSONDecodeError:
                params = {}

            try:
                if params.get("hue") is not None:
                    nanoleaf.set_hue(params["hue"])
                if params.get("saturation") is not None:
                    nanoleaf.set_saturation(params["saturation"])
                if params.get("brightness") is not None:
                    nanoleaf.set_brightness(params["brightness"])
                result = "Nanoleaf updated."
            except Exception as e:
                result = f"Error: {e}"

            self.log(result)
            return self._call_model(user_text, result)

        if action == "briefing.morning":
            self.log("Morning briefing started.")
            try:
                # Gather briefing data
                data = gather_briefing_data()
                briefing_text = format_briefing_for_llm(data)
            except Exception as e:
                self.log(f"Briefing data error: {e}")
                briefing_text = "(Could not gather briefing data.)"

            # Set morning light mood
            try:
                mood = get_time_of_day_light_mood("morning")
                if mood.get("govee"):
                    govee_style = mood["govee"]
                    set_lights_style(govee_style)
                if mood.get("nanoleaf"):
                    nl_style = mood["nanoleaf"]
                    if nl_style.get("color_hex"):
                        h = nl_style["color_hex"]
                        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
                        nanoleaf.set_color_rgb(r, g, b)
                    if nl_style.get("brightness") is not None:
                        nanoleaf.set_brightness(nl_style["brightness"])
                    self.log("Morning: Lights set to energizing mood.")
            except Exception as e:
                self.log(f"Morning lights set failed: {e}")

            return self._call_model(
                user_text,
                None,
                extra_note=(
                    "System note: The user is starting their day. The lights have been set to an "
                    "energizing mood. Here is their morning briefing:\n\n"
                    + briefing_text
                    + "\n\nGive a warm, brief good morning message (2–3 sentences). "
                    "Highlight any important emails or calendar items if relevant.\n\n"
                ),
            )

        if action == "briefing.goodnight":
            self.log("Goodnight briefing started.")
            # Set goodnight light mood
            try:
                mood = get_goodnight_light_mood()
                if mood.get("govee"):
                    govee_style = mood["govee"]
                    set_lights_style(govee_style)
                if mood.get("nanoleaf"):
                    nl_style = mood["nanoleaf"]
                    if nl_style.get("color_hex"):
                        h = nl_style["color_hex"]
                        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
                        nanoleaf.set_color_rgb(r, g, b)
                    if nl_style.get("brightness") is not None:
                        nanoleaf.set_brightness(nl_style["brightness"])
                    self.log("Goodnight: Nanoleaf dimmed.")
            except Exception as e:
                self.log(f"Goodnight Nanoleaf set failed: {e}")
            return self._call_model(
                user_text,
                None,
                extra_note=(
                    "System note: The user is winding down for the night. The lights have been "
                    "dimmed to a warm sleepy glow. Give a short, calming goodnight message. "
                    "Be warm and brief (1–3 sentences). No need to mention email or tasks.\n\n"
                ),
            )

        if action == "email.search":
            # Ask the model to extract the search query
            query_prompt = f"""Extract the email search query from this message: "{user_text}"
Respond with ONLY the keyword or label (no explanation). Examples: "contracts", "reviews", "feature"."""
            query = ask_lmstudio(query_prompt).strip().lower()

            if not query:
                result = "No search query extracted."
            else:
                try:
                    emails = search_gmail(query)
                    if emails:
                        result = f"Found {len(emails)} email(s) about '{query}':\n"
                        for email in emails[:5]:  # Show first 5
                            result += f"- {email.get('from', 'Unknown')}: {email.get('subject', '(no subject)')}\n"
                    else:
                        result = f"No emails found about '{query}'."
                except GmailClientError as e:
                    result = f"Gmail error: {e}"

            self.log(result)
            return self._call_model(user_text, result)

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