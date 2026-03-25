"""
Assistant tool definitions and prompt helpers.

This is extracted from `assistant_engine.py` to keep that module smaller and easier
to maintain, without changing behavior.
"""

from __future__ import annotations

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
        "name": "nanoleaf.set_scene",
        "description": "Set ONLY the Nanoleaf panels to a predefined SCENE from the list (e.g. Romantic, Northern Lights, Inner Peace). Use when the user asks for a mood that fits a scene name. Do NOT use for: pulse, 'add a pulse', breathing, 'custom', 'without using scenes', 'make up your own'—use nanoleaf.custom for those.",
        "params": [{"name": "description", "type": "string", "description": "The mood that should match a scene name (e.g. romantic, peaceful)"}],
    },
    {
        "name": "nanoleaf.custom",
        "description": "Set BOTH Govee and Nanoleaf to a static color and brightness, or pick a built-in Nanoleaf scene for pulse-like looks. Use for STATIC or SOLID color (e.g. 'static purple', 'solid red', 'no animation'). Do NOT use for 'create animation', 'custom animation', 'flowing', 'rainbow cycle', or colors moving across panels—those require nanoleaf.create_animation. Do NOT use for a mood that matches a predefined scene name—use nanoleaf.set_scene instead.",
        "params": [{"name": "description", "type": "string", "description": "The full request (e.g. static purple, no animation blue, strong pulse, custom somber)"}],
    },
    {
        "name": "nanoleaf.create_animation",
        "description": "CREATE a flowing/animated color effect on Nanoleaf (colors cycle across panels). Use for 'create animation', 'custom animation', 'new animation', 'rainbow', 'flow', 'cycle colors', 'red and blue alternating', any request for moving/changing colors over time. Pick 2–6 hex colors in params.colors (e.g. #FF0000, #0000FF); if unsure, use red+blue or a rainbow set. Do NOT use for solid/static single color—use nanoleaf.custom.",
        "params": [
            {"name": "animation_type", "type": "string", "description": "One of: flow"},
            {"name": "colors", "type": "array", "description": 'List of hex color strings, e.g. ["#FF0000", "#00FF00", "#0000FF"]. Need at least 2 for flow.'},
            {"name": "speed", "type": "number", "description": "Optional. Transition speed 0.5–5 (seconds). Default 1."},
        ],
    },
    {
        "name": "nanoleaf.set_brightness",
        "description": "Set ONLY the Nanoleaf panels' brightness (no scene change, no color change, no Govee). Use when the user asks to make the Nanoleaf dimmer, brighter, or set to a specific brightness level. Do NOT use lights.set_scene for this—that would change the scene and Govee.",
        "params": [{"name": "description", "type": "string", "description": "e.g. dimmer, brighter, 50%, half brightness"}],
    },
    {
        "name": "nanoleaf.set_state",
        "description": "Turn ONLY the Nanoleaf panels on or off. Use when the user says 'turn the nanoleaf off', 'nanoleaf on', 'turn off the nanoleaf', etc. Does NOT change Govee. Do NOT use nanoleaf.custom for power off/on—use this tool.",
        "params": [{"name": "state", "type": "string", "description": "on or off"}],
    },
    {
        "name": "memory.remember",
        "description": "Remember a simple mapping the user defines, like 'when I say writing mode, I mean dim orange lights'. Use ONLY when the user explicitly asks you to remember something for the future ('remember that X means Y', 'from now on when I say X do Y').",
        "params": [
            {"name": "key", "type": "string", "description": "The phrase to remember (e.g. writing mode)"},
            {"name": "value", "type": "string", "description": "What it should mean/expand to (e.g. dim orange nanoleaf and warm white govee at 35% brightness)"},
        ],
    },
]

VALID_ACTIONS = {t["name"] for t in TOOLS}


def _format_tools_for_prompt() -> str:
    """Format TOOLS into a string for the LLM prompt."""
    def _param_to_str(p: dict) -> str:
        base = f'{p["name"]} ({p["type"]})'
        desc = p.get("description")
        return base + (f": {desc}" if desc else "")

    lines = []
    for t in TOOLS:
        params_str = ""
        if t["params"]:
            params_str = " Parameters: " + ", ".join(_param_to_str(p) for p in t["params"])
        lines.append(f'- {t["name"]}: {t["description"]}{params_str}')
    return "\n".join(lines)

