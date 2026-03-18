"""
Headless assistant engine: route user message, run tools, call LLM, return reply string.
Used by both the desktop GUI and the FastAPI server.
"""

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
        "description": "Turn BOTH Govee and Nanoleaf lights on, off, or set Govee to automatic mode. Use only when the user explicitly says on, off, or auto/automatic\u2014not for moods or scenes.",
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
        "description": "Search the user\u2019s Gmail inbox. Use when they ask about email, inbox, messages, or things that would be in email (e.g. literary magazine submissions, acceptances).",
        "params": [
            {"name": "query", "type": "string", "description": "Short search terms (plain words, no operators)"},
            {"name": "scope", "type": "string", "description": "unread or all"},
            {"name": "result_type", "type": "string", "description": "count or list"},
            {"name": "category", "type": "string", "description": "(optional) updates, primary, promotions, social, forums"},
            {"name": "broad_search_terms", "type": "array", "description": "(optional) list of terms for list searches to widen results"},
        ],
    },
    {
        "name": "nanoleaf.set_scene",
        "description": "Set ONLY the Nanoleaf panels to a predefined SCENE from the list (e.g. Romantic, Northern Lights, Inner Peace). Use when the user asks for a mood that fits a scene name. Do NOT use for: pulse, 'add a pulse', breathing, 'custom', 'without using scenes', 'make up your own'\u2014use nanoleaf.custom for those.",
        "params": [{"name": "description", "type": "string", "description": "The mood that should match a scene name (e.g. romantic, peaceful)"}],
    },
    {
        "name": "nanoleaf.custom",
        "description": "Set BOTH Govee and Nanoleaf to a static color and brightness, or a pulse/custom mood. Use when the user wants a STATIC or SOLID color with NO animation (e.g. 'static purple', 'no animation just blue', 'solid red', 'make them purple but not animated'). Also use for pulse, 'add a pulse', rhythm, or 'make up your own settings'. Interpret their words: if they say static, no animation, solid, or don\u2019t want movement \u2192 this tool sets a single static color; if they want a flowing/animated effect \u2192 use nanoleaf.create_animation instead. Do NOT use for a mood that matches a scene name\u2014use nanoleaf.set_scene instead.",
        "params": [{"name": "description", "type": "string", "description": "The full request (e.g. static purple, no animation blue, strong pulse, custom somber)"}],
    },
    {
        "name": "nanoleaf.create_animation",
        "description": "CREATE a flowing/animated color effect on the Nanoleaf panels (colors cycle or flow). Use ONLY when the user clearly wants movement/animation: e.g. 'create an animation', 'flowing colors', 'make them cycle through red and blue', 'new animation'. Do NOT use when they want a static, solid, or non-animated color\u2014use nanoleaf.custom for that. Pick 2\u20136 colors and optional speed.",
        "params": [
            {"name": "animation_type", "type": "string", "description": "One of: flow"},
            {"name": "colors", "type": "array", "description": "List of hex color strings, e.g. [\"#FF0000\", \"#00FF00\", \"#0000FF\"]. Need at least 2 for flow."},
            {"name": "speed", "type": "number", "description": "(optional) Transition speed 0.5\u20135 (seconds). Default 1."},
        ],
    },
    {
        "name": "nanoleaf.set_brightness",
        "description": "Set ONLY the Nanoleaf panels' brightness (no scene change, no color change, no Govee). Use when the user asks to make the Nanoleaf dimmer, brighter, or set to a specific brightness level. Do NOT use lights.set_scene for this\u2014that would change the scene and Govee.",
        "params": [{"name": "description", "type": "string", "description": "e.g. dimmer, brighter, 50%, half brightness"}],
    },
    {
        "name": "nanoleaf.set_state",
        "description": "Turn ONLY the Nanoleaf panels on or off. Use when the user says 'turn the nanoleaf off', 'nanoleaf on', 'turn off the nanoleaf', etc. Does NOT change Govee. Do NOT use nanoleaf.custom for power off/on\u2014use this tool.",
        "params": [{"name": "state", "type": "string", "description": "on or off"}],
    },
    {
        "name": "memory.remember",
        "description": "Remember a simple mapping the user defines, like 'when I say writing mode, I mean dim orange lights'. Use ONLY when the user explicitly asks you to remember something for the future ('remember that X means Y', 'from now on when I say X do Y').",
        "params": [
            {"name": "key", "type": "string", "description": "The phrase to remember (e.g. writing mode)"},
            {
                "name": "value",
                "type": "string",
                "description": "What it should mean/expand to (e.g. dim orange nanoleaf and warm white govee at 35% brightness)",
            },
        ],
    },
    {
        "name": "reminder.set",
        "description": "Set a timed reminder. Use when the user asks to be reminded of something in a certain amount of time (e.g. 'remind me in 30 minutes to check the oven', 'set a timer for 10 minutes', 'remind me in an hour to call mom'). The Nanoleaf will flash gold when it fires.",
        "params": [
            {"name": "message", "type": "string", "description": "What to remind them about (e.g. check the oven, take a break, call mom)"},
            {"name": "minutes", "type": "number", "description": "How many minutes from now (e.g. 30, 5, 60, 0.5 for 30 seconds)"},
        ],
    },
    {
        "name": "reminder.list",
        "description": "List active (pending) reminders. Use when the user asks what reminders they have, 'any reminders?', 'what timers are running?', or 'show my reminders'.",
        "params": [],
    },
    {
        "name": "reminder.cancel",
        "description": "Cancel a pending reminder. Use when the user asks to cancel or remove a reminder (e.g. 'cancel reminder 1', 'never mind about that reminder', 'cancel my timer').",
        "params": [
            {"name": "reminder_id", "type": "number", "description": "The reminder ID to cancel (shown in reminder.list output)"},
        ],
    },
]

VALID_ACTIONS = {t["name"] for t in TOOLS}
