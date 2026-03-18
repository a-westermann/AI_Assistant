import json
import os
import re
import subprocess
import threading
from typing import Any, Callable, Optional

from llm import ask_lmstudio
from lights_client import (
    get_lights_state,
    toggle_all_lights,
    LightsClientError,
    set_lights_auto,
    set_lights_style,
)
from nanoleaf import nanoleaf
from gmail_client import search_gmail, GmailClientError
from user_memory import remember_alias, resolve_alias
from weather_client import get_current_weather_summary, WeatherClientError
from reminder_engine import get_reminder_engine

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
        "description": "Set Nanoleaf to a specific scene or color. Use only when user mentions a mood, scene, or color preference (not brightness/on/off).",
        "params": [{"name": "scene", "type": "string", "description": "Scene name or color description, e.g. 'blue', 'warm', 'energetic'"}],
    },
    {
        "name": "plex.sync",
        "description": "Sync the Plex library from disk (this may take 30+ seconds). Use when the user asks to refresh or update the Plex library.",
        "params": [],
    },
    {
        "name": "weather.current",
        "description": "Get current weather summary. Use when the user asks about the weather.",
        "params": [],
    },
    {
        "name": "gmail.search",
        "description": "Search Gmail for emails. Use when the user asks to find, check, or search emails.",
        "params": [{"name": "query", "type": "string", "description": "Gmail search query, e.g. 'from:alice subject:deadline'"}],
    },
    {
        "name": "memory.remember",
        "description": "Remember something about the user or add an alias. Use when the user tells you something personal or asks you to remember something.",
        "params": [{"name": "fact", "type": "string", "description": "The fact or alias to remember, e.g. 'Alice is my sister'"}],
    },
    {
        "name": "reminder.add",
        "description": "Set a reminder to notify the user at a specific time. Use when the user asks to set a reminder, timer, or alarm.",
        "params": [
            {"name": "message", "type": "string", "description": "What to remind about"},
            {"name": "minutes", "type": "number", "description": "Minutes from now to fire the reminder"},
        ],
    },
    {
        "name": "reminder.cancel",
        "description": "Cancel a pending reminder by ID. Use when the user wants to cancel a reminder they set earlier.",
        "params": [{"name": "reminder_id", "type": "number", "description": "ID of the reminder to cancel"}],
    },
    {
        "name": "reminder.list",
        "description": "List all active reminders. Use when the user asks what reminders they have set.",
        "params": [],
    },
]


def handle_message(message: str, log_fn: Optional[Callable[[str], None]] = None) -> str:
    """Route and process a user message: choose tool, execute, get LLM response."""
    log = log_fn or (lambda msg: None)

    # Build a prompt for the LLM to choose a tool.
    tool_descriptions = "\n".join(
        f"  - {t['name']}: {t['description']}"
        for t in TOOLS
    )

    routing_prompt = (
        "You are a helpful assistant. Based on the user's message, choose ONE tool from the list below. "
        "Reply with ONLY the tool name (nothing else), or 'none' if no tool applies.\n\n"
        f"Available tools:\n{tool_descriptions}\n\n"
        f"User message: {message!r}"
    )

    routing_response = ask_lmstudio(routing_prompt)
    tool_name = (routing_response.get("output") or [{}])[0].get("content", "").strip().lower()

    log(f"Routing: message={message!r} -> tool={tool_name}")

    result = None
    tool_output = ""

    # Execute the chosen tool.
    if tool_name == "lights.set_state":
        state_match = re.search(r"\b(on|off|auto)\b", message, re.IGNORECASE)
        state = state_match.group(1).lower() if state_match else None
        if state:
            try:
                if state == "on":
                    toggle_all_lights(on=True)
                    tool_output = "Lights turned on."
                elif state == "off":
                    toggle_all_lights(on=False)
                    tool_output = "Lights turned off."
                elif state == "auto":
                    set_lights_auto()
                    tool_output = "Lights set to automatic."
            except LightsClientError as e:
                tool_output = f"Error controlling lights: {e}"
        else:
            tool_output = "Could not determine light state from message."

    elif tool_name == "lights.set_scene":
        scene_match = re.search(r"(?:set|change|go to|make it|switch to)\s+([a-z\s]+?)(?:\.|$|\?)", message, re.IGNORECASE)
        scene = scene_match.group(1).strip() if scene_match else None
        if scene:
            try:
                set_lights_style(scene)
                tool_output = f"Lights set to {scene}."
            except LightsClientError as e:
                tool_output = f"Error setting light scene: {e}"
        else:
            tool_output = "Could not determine desired scene from message."

    elif tool_name == "plex.sync":
        try:
            log("Plex sync starting...")
            result = subprocess.run(
                [PLEX_SYNC_PY, PLEX_SYNC_MAIN],
                capture_output=True,
                text=True,
                timeout=120,
            )
            tool_output = result.stdout or result.stderr or "Plex sync complete."
        except Exception as e:
            tool_output = f"Error syncing Plex: {e}"

    elif tool_name == "weather.current":
        try:
            tool_output = get_current_weather_summary()
        except WeatherClientError as e:
            tool_output = f"Error fetching weather: {e}"

    elif tool_name == "gmail.search":
        query_match = re.search(r"(?:search|find|check)\s+(?:for\s+)?(.+?)(?:\?|$)", message, re.IGNORECASE)
        query = query_match.group(1).strip() if query_match else None
        if not query:
            for term in _GENERIC_BROAD_TERMS:
                if term.lower() in message.lower():
                    query = term
                    break
        if query:
            try:
                results = search_gmail(query)
                tool_output = results or f"No emails found for '{query}'."
            except GmailClientError as e:
                tool_output = f"Error searching Gmail: {e}"
        else:
            tool_output = "Could not determine search query."

    elif tool_name == "memory.remember":
        fact_match = re.search(r"(?:remember|my|is|that\s+)(.+?)(?:\.|$)", message, re.IGNORECASE)
        fact = fact_match.group(1).strip() if fact_match else message
        try:
            remember_alias(fact)
            tool_output = f"Remembered: {fact}"
        except Exception as e:
            tool_output = f"Error remembering: {e}"

    elif tool_name == "reminder.add":
        minutes_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)", message, re.IGNORECASE)
        minutes = float(minutes_match.group(1)) if minutes_match else 10
        reminder_msg = re.sub(r"set\s+a?\s+reminder|in\s+\d+\s*(?:minutes?|mins?|m)", "", message, flags=re.IGNORECASE).strip()
        try:
            engine = get_reminder_engine(log_fn=log)
            reminder = engine.add(reminder_msg or "reminder", minutes)
            tool_output = f"Reminder set for {minutes} minutes: {reminder_msg or 'reminder'}"
        except Exception as e:
            tool_output = f"Error setting reminder: {e}"

    elif tool_name == "reminder.list":
        try:
            engine = get_reminder_engine(log_fn=log)
            active = engine.list_active()
            if active:
                tool_output = "Active reminders:\n" + "\n".join(
                    f"  #{r['id']}: {r['message']} (in {r['remaining_seconds']//60}m)"
                    for r in active
                )
            else:
                tool_output = "No active reminders."
        except Exception as e:
            tool_output = f"Error listing reminders: {e}"

    elif tool_name == "reminder.cancel":
        reminder_id_match = re.search(r"#?(\d+)", message)
        reminder_id = int(reminder_id_match.group(1)) if reminder_id_match else None
        if reminder_id:
            try:
                engine = get_reminder_engine(log_fn=log)
                success = engine.cancel(reminder_id)
                tool_output = f"Reminder #{reminder_id} cancelled." if success else f"Reminder #{reminder_id} not found."
            except Exception as e:
                tool_output = f"Error cancelling reminder: {e}"
        else:
            tool_output = "Could not find reminder ID in message."

    # Now ask the LLM to generate a final response, given tool_output.
    if tool_output:
        final_prompt = (
            f"Based on the user's request and the tool result below, provide a helpful, natural response. "
            f"Be concise (1–2 sentences unless more detail is needed).\n\n"
            f"User: {message}\n"
            f"Tool result: {tool_output}"
        )
    else:
        final_prompt = (
            f"The user sent a message. Respond helpfully and naturally. Be concise.\n\n"
            f"User: {message}"
        )

    final_response = ask_lmstudio(final_prompt)
    reply = (final_response.get("output") or [{}])[0].get("content", "").strip()

    log(f"Response: {reply!r}")
    return reply or "I'm not sure how to respond."


if __name__ == "__main__":
    msg = input("User: ")
    reply = handle_message(msg, log_fn=print)
    print("Reply:", reply)