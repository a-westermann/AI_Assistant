"""
Headless assistant engine: route user message, run tools, call LLM, return reply string.
Used by both the desktop GUI and the FastAPI server.
"""

import json
import os
import re
import subprocess
import threading
import time
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
from weather_client import (
    get_current_weather_summary,
    get_day_weather_forecast_summary,
    WeatherClientError,
)

# Plex sync (same as chat_gui; override with PLEX_SYNC_DIR env if needed)
PLEX_SYNC_DIR = os.environ.get("PLEX_SYNC_DIR", r"H:\Coding\Python Projects\plex_sync")
PLEX_SYNC_PY = os.path.join(PLEX_SYNC_DIR, ".venv", "Scripts", "python.exe")
PLEX_SYNC_MAIN = os.path.join(PLEX_SYNC_DIR, "main.py")
TOOLS_HELP_FILE = os.path.join(os.path.dirname(__file__), "tools.txt")

# Gmail broad-term filter for list searches
_GENERIC_BROAD_TERMS = {"review", "feature", "contract"}

from assistant_engine_tools import TOOLS, VALID_ACTIONS, _format_tools_for_prompt
from assistant_engine_lighting import try_handle_lighting_action
from auto_lighting_sync import start_auto_lighting_sync, stop_auto_lighting_sync


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
        self._profile_active: bool = False
        self._active_user_name: str = "Andrew"

    def _is_lighting_related(self, text: str) -> bool:
        t = (text or "").lower()
        # Treat good morning / good night as lighting routines
        if "good morning" in t or "good night" in t:
            return True
        return any(
            w in t
            for w in (
                "light",
                "lights",
                "nanoleaf",
                "govee",
                "scene",
                "brightness",
                "dim",
                "bright",
                "cozy",
                "warm",
            )
        )

    def _is_memory_request(self, text: str) -> bool:
        """Detect explicit user instructions to store an alias mapping."""
        t = (text or "").lower()
        if "remember" not in t:
            return False
        return any(
            s in t
            for s in (
                "remember that",
                "remember when",
                "from now on",
                "when i say",
                "i mean",
                "means",
                "do this when",
            )
        )

    def _is_help_request(self, text: str) -> bool:
        """Detect explicit requests for capability/help summary."""
        raw = (text or "").strip().lower()
        if not raw:
            return False
        # Normalize punctuation/spacing so voice transcripts like "Help." still match.
        t = re.sub(r"[^\w\s/]", "", raw)
        t = re.sub(r"\s+", " ", t).strip()
        return t in {
            "help",
            "/help",
            "what can you do",
            "what can galadrial do",
            "what tools do you have",
            "show tools",
            "list tools",
            "show commands",
            "list commands",
            "help me",
            "can you help",
            "can you help me",
        }

    def _load_help_tools(self) -> list[str]:
        """Load user-facing tool/capability lines from tools.txt."""
        try:
            if not os.path.isfile(TOOLS_HELP_FILE):
                return []
            with open(TOOLS_HELP_FILE, "r", encoding="utf-8") as f:
                items = [line.strip() for line in f.readlines() if line.strip()]
            return items
        except Exception as e:
            self.log(f"Help tools load failed: {e}")
            return []

    def _is_weather_query(self, text: str) -> bool:
        """Detect whether the user is asking for current weather information."""
        t = (text or "").lower()
        # "forecast"/"today"/"rain" are handled by the forecast function.
        return any(k in t for k in ("weather", "temperature", "temp", "degree", "degrees", "right now"))

    def _is_weather_forecast_query(self, text: str) -> bool:
        """Detect day-level weather forecast questions."""
        t = (text or "").lower()
        # Never classify lighting/animation requests as weather just because
        # they contain substrings like "rainbow".
        lighting_terms = (
            "nanoleaf",
            "govee",
            "govee lights",
            "lights",
            "light",
            "animation",
            "scene",
            "brightness",
        )
        if any(x in t for x in lighting_terms):
            return False

        # Weather forecast signals.
        if any(k in t for k in ("forecast", "today", "for the day", "precipitation", "precip")):
            return True
        if any(k in t for k in ("high", "peak", "tonight")):
            return True
        # "rain" should be a standalone word to avoid matching "rainbow".
        if re.search(r"\brain\b", t):
            return True
        return False

    def _is_brightness_status_query(self, text: str) -> bool:
        """
        Detect queries like:
        - "what is the current brightness of govee lights?"
        - "how bright are the govee lights right now?"
        - "govee brightness right now"

        Excludes commands that change brightness (set/make/dim/brighter/etc.).
        """
        t = (text or "").lower()
        if "brightness" not in t and "how bright" not in t and "bright" not in t:
            return False
        # Must be about asking, not setting.
        set_triggers = ("set", "make", "dim", "dimmer", "brighter", "brighten", "increase", "decrease", "%")
        if any(s in t for s in set_triggers):
            return False
        # Prefer Govee mentions if present, but don't require it.
        return "govee" in t or "govee lights" in t or "lights" in t or "bulb" in t

    def _is_lights_set_to_query(self, text: str) -> bool:
        """
        Detect questions like "what are the lights set to" / "what are the lights set at"
        so we can answer from the Govee status endpoint instead of the lighting pipeline.
        """
        t = (text or "").lower().strip()
        if not any(w in t for w in ("lights set", "lights are set", "lights set to", "lights set at", "what are the lights", "what is the color", "what color", "what is the temperature")):
            return False
        # Exclude explicit commands that would change lights.
        change_triggers = ("set", "make", "turn", "dim", "bright", "romantic", "cozy", "pulse", "animation", "scene", "flow", "create")
        # The query will contain "set" but in a "what ... set to" context; exclude if it starts as an imperative.
        if any(t.startswith(p) for p in ("set ", "make ", "turn ", "dim ", "bright ", "make the ", "change ")):
            return False
        return True

    def _is_too_dark_to_get_bright(self, text: str) -> bool:
        """
        Detect requests that mean "turn the lights brighter" / "it's dark in here".
        We only handle brightness increases (not color changes).
        """
        t = (text or "").lower()
        # If the user is explicitly asking to make things dimmer, don't treat it as "it's too dark".
        if any(w in t for w in ("dimmer", "make it dimmer", "dim it", "darken", "darker")):
            return False

        if not (
            "too dark" in t
            or "dark in here" in t
            or "dark in " in t
            or "not bright" in t
            or "not bright enough" in t
            or "low light" in t
            or re.search(r"\btoo dim\b", t)
            or re.search(r"\bdim in (here|the room)\b", t)
        ):
            return False

        # If the user also mentions an explicit color/scene/effect, don't force brightness-only behavior.
        color_words = (
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "purple",
            "pink",
            "cyan",
            "teal",
            "amber",
            "indigo",
            "violet",
            "gold",
            "white",
            "black",
        )
        if any(c in t for c in color_words):
            return False
        if any(s in t for s in ("scene", "romantic", "cozy", "warm", "cool", "pulse", "animation", "flow", "rainbow")):
            return False
        return True

    def _should_skip_tool_router(self, text: str) -> bool:
        """
        If the message is clearly just general chat (no lights/gmail/plex/memory actions),
        skip the router LLM call to reduce latency.
        """
        t = (text or "").lower()
        tool_triggers = (
            "lights",
            "light",
            "nanoleaf",
            "govee",
            "scene",
            "brightness",
            "dim",
            "bright",
            "cozy",
            "warm",
            "sexy",
            "romantic",
            "pulse",
            "animation",
            "flow",
            "create",
            "email",
            "gmail",
            "inbox",
            "plex",
            "sync",
            "remember",
            "weather",
            "forecast",
            "temperature",
        )
        return not any(k in t for k in tool_triggers)

    def _try_parse_remember_mapping(self, user_text: str) -> tuple[str, str] | None:
        """
        Parse mappings like:
        - "remember that X means Y"
        - "remember when I say X I mean Y"
        - "from now on when i say X i mean Y"
        """
        if not self._is_memory_request(user_text):
            return None

        original = user_text or ""
        lower = original.lower()

        # Case 1: "remember that X means Y"
        if "remember that" in lower:
            start = lower.index("remember that") + len("remember that")
            # Prefer " means " separator; fall back to " i mean "
            means_idx = lower.find(" means ", start)
            imean_idx = lower.find(" i mean ", start)
            if means_idx == -1:
                means_idx = imean_idx
                sep_len = len(" i mean ")
            else:
                sep_len = len(" means ")
            if means_idx != -1:
                key = original[start:means_idx].strip().strip("\"'")
                value = original[means_idx + sep_len :].strip().strip()
                value = value.strip().strip(".")
                if value.lower().startswith("to "):
                    value = value[3:].strip()
                if key and value:
                    return key, value

        # Case 2: "remember when i say X i mean Y"
        if "when i say" in lower:
            start = lower.index("when i say") + len("when i say")
            candidates = [" i mean ", " i mean to ", " means ", " mean ", " i.e. "]
            end = -1
            end_sep = ""
            for c in candidates:
                idx = lower.find(c, start)
                if idx != -1 and (end == -1 or idx < end):
                    end = idx
                    end_sep = c
            if end != -1:
                key = original[start:end].strip().strip("\"'")
                value = original[end + len(end_sep) :].strip().strip()
                value = value.strip().strip(".")
                if value.lower().startswith("to "):
                    value = value[3:].strip()
                if key and value:
                    return key, value

        return None

    def _parse_lighting_plan(self, user_text: str, scene_list: list[str]) -> dict | None:
        """
        LLM semantic parser: user text -> structured plan JSON (NO tool choice).
        The plan must be deterministic enough for code to execute.
        """
        scenes_blob = ", ".join(scene_list) if scene_list else "(none)"
        prompt = f"""You are Galadrial. Parse the user's message into a structured LIGHTING PLAN JSON.
Do NOT choose code tools or APIs. Do NOT mention tools. Only describe what should happen.

Available Nanoleaf predefined scenes (must pick from this exact list if user asks for a scene): {scenes_blob}

Schema:
{{
  "targets": ["nanoleaf", "govee"],          # affected devices
  "exclude": ["govee"],                     # devices explicitly excluded
  "actions": [
    {{
      "device": "nanoleaf" | "govee",
      "type": "power" | "static_color" | "brightness" | "animation" | "scene",
      "state": "on" | "off" | null,         # for power
      "color_hex": "#RRGGBB" | null,        # for static_color
      "colors_hex": ["#RRGGBB", ...] | null,# for animation (2-6 colors)
      "scene_name": "Scene Name" | null,    # for scene (must be from list)
      "brightness": 0-100 | null,           # for static_color/brightness
      "speed": 0.5-5 | null,                # for animation
      "animation": true | false | null      # if user specifies no animation
    }}
  ]
}}

Hard rules:
- If user says static/solid/no animation/no movement -> must produce type \"static_color\" (animation=false) NOT animation.
- If user says animation/flow/cycle/pulse/moving -> type must be \"animation\" (animation=true).
- If user asks to choose from predefined scenes / a scene / an effect -> type must be \"scene\" with scene_name chosen from the available list.
- If user says \"just/only nanoleaf\" -> targets must be [\"nanoleaf\"], exclude must include \"govee\".
  The ONLY allowed govee action in that case is power off (if explicitly requested).
- If user says \"turn govee off\" -> include govee power off.
- If user mentions a color name, convert it to a reasonable hex (green=#00FF00, orange=#FFA500, purple=#800080, etc.).
- If user mentions a percent, set brightness to that number. If user says dim/dimmer/dark -> 20-40. bright/brighter/max -> 80-100.
- If brightness is not mentioned, brightness may be null.

User: \"{user_text}\"

Reply with JSON only (no markdown)."""
        try:
            response = ask_lmstudio(prompt)
            raw = (response.get("output") or [{}])[0].get("content", "").strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1].strip()
            plan = json.loads(raw)
            return plan if isinstance(plan, dict) else None
        except Exception as e:
            self.log(f"Lighting plan parse failed: {e}")
            return None

    def _validate_lighting_plan(self, plan: dict, user_text: str, scene_list: list[str]) -> dict:
        """Clamp and enforce safety constraints; returns a cleaned plan dict."""
        cleaned: dict = {"targets": [], "exclude": [], "actions": []}
        targets = plan.get("targets") if isinstance(plan.get("targets"), list) else []
        exclude = plan.get("exclude") if isinstance(plan.get("exclude"), list) else []
        cleaned["targets"] = [str(x).lower() for x in targets if str(x).lower() in ("nanoleaf", "govee")]
        cleaned["exclude"] = [str(x).lower() for x in exclude if str(x).lower() in ("nanoleaf", "govee")]
        actions = plan.get("actions") if isinstance(plan.get("actions"), list) else []
        for a in actions:
            if not isinstance(a, dict):
                continue
            device = str(a.get("device") or "").lower()
            atype = str(a.get("type") or "").lower()
            if device not in ("nanoleaf", "govee"):
                continue
            if atype not in ("power", "static_color", "brightness", "animation", "scene"):
                continue
            out = {"device": device, "type": atype}
            if atype == "power":
                state = str(a.get("state") or "").lower()
                out["state"] = "on" if state == "on" else "off"
            if atype == "brightness":
                b = a.get("brightness")
                try:
                    out["brightness"] = max(0, min(100, int(b)))
                except Exception:
                    out["brightness"] = None
            if atype == "static_color":
                ch = a.get("color_hex")
                if isinstance(ch, str) and ch.strip().startswith("#") and len(ch.strip()) == 7:
                    out["color_hex"] = ch.strip()
                else:
                    out["color_hex"] = None
                b = a.get("brightness")
                try:
                    out["brightness"] = max(0, min(100, int(b))) if b is not None else None
                except Exception:
                    out["brightness"] = None
            if atype == "animation":
                cols = a.get("colors_hex")
                if isinstance(cols, list):
                    out_cols = []
                    for c in cols:
                        s = str(c).strip()
                        if s.startswith("#") and len(s) == 7:
                            out_cols.append(s)
                    out["colors_hex"] = out_cols[:6]
                else:
                    out["colors_hex"] = []
                sp = a.get("speed")
                try:
                    out["speed"] = max(0.5, min(5.0, float(sp))) if sp is not None else 1.0
                except Exception:
                    out["speed"] = 1.0
            if atype == "scene":
                name = a.get("scene_name")
                if isinstance(name, str):
                    scene = name.strip()
                    # Only allow selecting scenes from scenes.txt (exact match)
                    out["scene_name"] = scene if scene in scene_list else None
                else:
                    out["scene_name"] = None
            cleaned["actions"].append(out)

        # Deterministic scope default: if user only mentioned nanoleaf (not govee, not "lights"),
        # never allow govee actions unless explicitly requested (e.g. "turn govee off").
        t = (user_text or "").lower()
        if "nanoleaf" in t and "govee" not in t and " lights" not in t and not t.startswith("lights"):
            cleaned["exclude"] = list(set(cleaned["exclude"] + ["govee"]))

        # Enforce: if govee excluded, only allow govee power off actions.
        if "govee" in cleaned["exclude"]:
            new_actions = []
            for a in cleaned["actions"]:
                if a["device"] != "govee":
                    new_actions.append(a)
                elif a["type"] == "power" and a.get("state") == "off":
                    new_actions.append(a)
            cleaned["actions"] = new_actions

        # Drop invalid scene actions (no matching scene_name).
        cleaned["actions"] = [
            a for a in cleaned["actions"] if not (a["type"] == "scene" and not a.get("scene_name"))
        ]

        return cleaned

    def _execute_lighting_plan(self, plan: dict) -> str:
        """Deterministically execute a validated lighting plan. Returns a short summary for system note."""
        actions = plan.get("actions") if isinstance(plan.get("actions"), list) else []
        notes: list[str] = []
        for a in actions:
            device = a.get("device")
            atype = a.get("type")
            if device == "govee":
                if atype == "power":
                    state = a.get("state", "off")
                    try:
                        toggle_all_lights("on" if state == "on" else "off")
                        notes.append(f"Govee turned {state}.")
                    except LightsClientError as e:
                        self.log(f"Govee power failed: {e}")
                        notes.append("Govee update failed.")
                elif atype == "static_color":
                    try:
                        set_lights_style(
                            state="on",
                            color_hex=a.get("color_hex"),
                            brightness=a.get("brightness"),
                        )
                        notes.append("Govee set to static color/brightness.")
                    except LightsClientError as e:
                        self.log(f"Govee static_color failed: {e}")
                        notes.append("Govee update failed.")
                elif atype == "brightness":
                    try:
                        set_lights_style(state="on", brightness=a.get("brightness"))
                        notes.append("Govee brightness updated.")
                    except LightsClientError as e:
                        self.log(f"Govee brightness failed: {e}")
                        notes.append("Govee update failed.")

            if device == "nanoleaf":
                if atype == "power":
                    state = a.get("state", "off")
                    try:
                        if state == "on":
                            nanoleaf.turn_on()
                        else:
                            nanoleaf.turn_off()
                        notes.append(f"Nanoleaf turned {state}.")
                    except Exception as e:
                        self.log(f"Nanoleaf power failed: {e}")
                        notes.append("Nanoleaf update failed.")
                elif atype == "brightness":
                    b = a.get("brightness")
                    try:
                        if b is not None:
                            nanoleaf.set_brightness(int(b))
                            notes.append("Nanoleaf brightness updated.")
                    except Exception as e:
                        self.log(f"Nanoleaf brightness failed: {e}")
                        notes.append("Nanoleaf update failed.")
                elif atype == "static_color":
                    ch = a.get("color_hex")
                    b = a.get("brightness")
                    try:
                        nanoleaf.turn_on()
                        if isinstance(ch, str) and ch.startswith("#") and len(ch) == 7:
                            r, g, bb = int(ch[1:3], 16), int(ch[3:5], 16), int(ch[5:7], 16)
                            nanoleaf.set_color_rgb(r, g, bb)
                        if b is not None:
                            nanoleaf.set_brightness(int(b))
                        notes.append("Nanoleaf set to static color/brightness.")
                    except Exception as e:
                        self.log(f"Nanoleaf static_color failed: {e}")
                        notes.append("Nanoleaf update failed.")
                elif atype == "animation":
                    cols = a.get("colors_hex") or []
                    speed = a.get("speed") or 1.0
                    try:
                        if isinstance(cols, list) and len(cols) >= 2:
                            rgb = [(int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)) for c in cols]
                            nanoleaf.create_flow_effect(rgb, float(speed))
                            notes.append("Nanoleaf animation applied.")
                    except Exception as e:
                        self.log(f"Nanoleaf animation failed: {e}")
                        notes.append("Nanoleaf update failed.")
                elif atype == "scene":
                    name = a.get("scene_name")
                    try:
                        if isinstance(name, str) and name.strip():
                            nanoleaf.turn_on()
                            nanoleaf.set_effect(name.strip())
                            notes.append(f"Nanoleaf scene set to '{name.strip()}'.")
                    except Exception as e:
                        self.log(f"Nanoleaf scene failed: {e}")
                        notes.append("Nanoleaf update failed.")

        return " ".join(notes) if notes else "No lighting changes applied."

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

    def _route_nanoleaf_power(self, user_text: str) -> dict | None:
        """If the user is asking to turn only the Nanoleaf on or off (no color/brightness), return nanoleaf.set_state. If they also want a color or brightness (e.g. 'turn on dim orange'), return None so the model chooses nanoleaf.custom."""
        t = user_text.lower().strip()
        if "nanoleaf" not in t:
            return None
        # If they're specifying a color or brightness level, use nanoleaf.custom instead so we set color+brightness
        if any(
            c in t for c in ("green", "purple", "blue", "red", "yellow", "orange", "pink", "cyan", "teal", "amber", "indigo", "violet", "gold", "white")
        ):
            return None
        if "dim" in t or "bright" in t or "%" in t:
            return None
        if " off" in t or t.endswith("off") or "turn off" in t:
            return {"action": "nanoleaf.set_state", "params": {"state": "off"}}
        if " on" in t or t.endswith("on") or "turn on" in t:
            return {"action": "nanoleaf.set_state", "params": {"state": "on"}}
        return None

    def _route_nanoleaf_brightness(self, user_text: str) -> dict | None:
        """If the user is asking only to change Nanoleaf brightness (no color), return nanoleaf.set_brightness. Do not use when they also ask for a color—let the model choose nanoleaf.custom."""
        t = user_text.lower().strip()
        if "nanoleaf" not in t and "nanoleaf lights" not in t:
            return None
        # If they're also specifying a color, don't force brightness-only—model should choose nanoleaf.custom
        if any(
            c in t for c in ("green", "purple", "blue", "red", "yellow", "orange", "pink", "white", "cyan", "teal", "amber", "indigo", "violet", "gold")
        ):
            return None
        if any(
            w in t for w in ("dimmer", "brighter", "brightness", "dim", "bright", "darker", "lighter", "%")
        ):
            return {
                "action": "nanoleaf.set_brightness",
                "params": {"description": user_text.strip()},
            }
        return None

    def route(self, user_text: str) -> dict:
        """Return {action, params} by asking the model to choose a tool from TOOLS based on the user's intent. No keyword routing."""
        prev_msg = self.last_user_message or ""
        prev_action = json.dumps(self.last_route) if self.last_route is not None else "null"
        # If user wants to change animation speed and we have a previous create_animation, reuse it with new speed so we actually apply
        speed_route = self._route_speed_adjust(user_text)
        if speed_route is not None:
            self.log("Router: using speed-adjust heuristic for nanoleaf animation.")
            return speed_route
        # "turn the nanoleaf off" / "nanoleaf on" → only Nanoleaf power, no Govee
        nanoleaf_power_route = self._route_nanoleaf_power(user_text)
        if nanoleaf_power_route is not None:
            self.log("Router: using nanoleaf power heuristic (on/off only).")
            return nanoleaf_power_route
        # "make the nanoleaf lights dimmer" / "nanoleaf brighter" → only Nanoleaf brightness, no scene or Govee
        nanoleaf_bright_route = self._route_nanoleaf_brightness(user_text)
        if nanoleaf_bright_route is not None:
            self.log("Router: using nanoleaf brightness heuristic.")
            return nanoleaf_bright_route
        tools_blob = _format_tools_for_prompt()

        tool_choice_prompt = f"""You are Galadrial, a desktop assistant. You have access to these tools. Interpret the user's message and choose ONE tool (or "none" only if they are clearly just chatting, not asking you to do something).

TOOLS:
{tools_blob}

Interpret intent from the actual request. Examples:
- "set the lights to romantic" / "romantic feel" / "make it cozy" / "add a gentle pulse" / "make them pulse faster and brighter" → lights.set_scene (params.description = their request or mood)
- "set the nanoleaf to romantic" (mood that matches a scene) → nanoleaf.set_scene
- "static purple" / "no animation, just blue" / "solid red" / "make them purple but not animated" → nanoleaf.custom (params.description = their full request). They want a single static color, no flowing animation.
- "create a new animation" / "flowing red and blue" / "make them cycle through colors" / "new animation with yellow" → nanoleaf.create_animation with colors and optional speed. They want movement/animation.
- "add a strong pulse" / "pulse" / "make up your own settings" → nanoleaf.custom.
- "make it faster" / "speed it up" / "slower" (when previous action was nanoleaf.create_animation) → nanoleaf.create_animation again with same colors and new speed.
- "turn the nanoleaf off" / "nanoleaf on" / "turn off the nanoleaf" → nanoleaf.set_state with state "off" or "on" (ONLY Nanoleaf; do NOT change Govee). Do NOT use nanoleaf.custom for this.
- "make the nanoleaf lights dimmer" / "nanoleaf brighter" / "set nanoleaf to 50%" → nanoleaf.set_brightness (ONLY Nanoleaf brightness; do NOT change scene or Govee). Do NOT use lights.set_scene for this.
- "turn lights on" / "lights off" (all lights) → lights.set_state with state "on" or "off"
- "are the lights on?" → lights.get_state
- "run Plex sync" → plex_sync.run
- "check my email for X" → gmail.search with query/scope/result_type
- "remember that writing mode means dim orange lights" → memory.remember (key: "writing mode", value: "dim orange lights")

If the user said "again" or "same thing", repeat the previous action. Do not choose "none" when the user is asking you to change the lights or set a mood—use lights.set_scene or nanoleaf.set_scene as appropriate. If they ask to "create a new animation" or "create an animation for the lights", always use nanoleaf.create_animation with colors (and optional speed)—never respond with a list of settings for them to enter in an app.

Previous user message: "{prev_msg}"
Previous action: {prev_action}

User message: "{user_text}"

Respond with JSON only:
{{"action": "<tool name from the list above>", "params": {{...}}}}
"""
        try:
            t0 = time.perf_counter() if self._profile_active else None
            response = ask_lmstudio(tool_choice_prompt)
            if self._profile_active and t0 is not None:
                self.log(f"PROFILE: router_model_call={time.perf_counter() - t0:.2f}s")
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
                self.log("Router: model returned unknown action; using fallback.")
                cleaned = self._heuristic_route(user_text)
            return cleaned
        except Exception as e:
            self.log(f"Router error in tool choice; using fallback. Details: {e}")
            cleaned = self._heuristic_route(user_text)
            return cleaned

    def _heuristic_route(self, user_text: str) -> dict:
        """Used only when the model fails or returns an unknown action. Minimal: repeat last action or none."""
        t = user_text.lower()
        if self.last_route is not None and any(
            phrase in t for phrase in ("again", "same thing", "do that", "check again")
        ):
            return self.last_route
        return {"action": "none", "params": {}}

    def _call_model(
        self,
        prompt: str,
        light_action: str | None = None,
        extra_note: str = "",
        *,
        compact_system: bool = False,
    ) -> str:
        """Build prompt, call LM Studio, return assistant reply text."""
        user_name = (self._active_user_name or "").strip() or "Guest"
        if compact_system:
            system_preamble = (
                "You are Galadrial, an AI assistant in a desktop GUI.\n"
                f"- Your user's name is {user_name}. Keep replies short and friendly.\n"
                "- IMPORTANT: Output plain ASCII text only. No markdown, no code fences, no emojis.\n"
                "- Safety: Do NOT invent facts about personal data. If you weren't given a tool/system"
                " result for a requested detail (like weather), say you don't know.\n"
                "- If you are given a system note saying the app set lights/weather, you may describe it.\n"
                "- If the user asks about D&D improv, direct them to http://localhost:8000/dnd.\n\n"
            )
        else:
            system_preamble = (
                "You are Galadrial, an AI assistant embedded in a desktop GUI.\n\n"
                f"- Your user's name is {user_name}. Give short, concise responses."
                " Always be warm and friendly.\n"
                "- IMPORTANT OUTPUT FORMAT: Output plain text only (no markdown, no code fences)."
                " Use ASCII characters only. Do not use emojis or decorative symbols,"
                " and avoid bullet points or special typographic punctuation."
                " When you write your reply, use normal sentences; do not start any line"
                " with '-' or '•'."
                "- The app can control my Govee lights via an API. When you see a system note telling"
                " you that the lights were set to a state, you may speak as if that action has already"
                " been performed.\n"
                "- The app may also display separate System messages with the results of tools"
                " (for example, Gmail searches, Plex sync, weather). Those System messages are the"
                " source of truth about my real data.\n"
                "- VERY IMPORTANT: Do NOT repeat System notes verbatim or list them like logs"
                " (e.g. 'Lights set:', 'Weather:', 'Quick news summary:'). Instead"
                " briefly and naturally incorporate only the important parts"
                " into your reply, or skip details I didn't explicitly ask for.\n\n"
                "CRITICAL SAFETY RULES:\n"
                "- Do NOT invent or guess specific facts about my personal data or file contents.\n"
                "- If you are not given an explicit System note or tool result that contains those"
                " facts, you must say that you don't know instead of making something up.\n"
                "- You may still answer general questions with your own knowledge, but never fabricate"
                " concrete details about my life or accounts.\n\n"
            "- Weather safety: If the user asks about weather (current or forecast), you may only state"
            " it when you were given a System note or tool result that contains the weather."
            " Otherwise, say you don't know.\n\n"
                "- The user has a D&D campaign folder (notes and maps). You do not have access to it"
                " from this chat. If they ask, say that the D&D improv feature does: when they open"
                " http://localhost:8000/dnd in a browser (with the API server running), they can"
                " record or paste conversation and click Get suggestion to get dialogue that uses"
                " that folder.\n\n"
            )
        action_note = ""
        if light_action in ("on", "off", "auto"):
            action_note = f"System note: The app has just set the lights to '{light_action}'.\n\n"
        if extra_note:
            action_note = action_note + extra_note
        full_prompt = f"{system_preamble}{action_note}User: {prompt}"
        try:
            t0 = time.perf_counter() if self._profile_active else None
            response = ask_lmstudio(full_prompt)
            if self._profile_active and t0 is not None:
                self.log(f"PROFILE: assistant_model_call={time.perf_counter() - t0:.2f}s")
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
                self.log(f"Email interpretation (no results) failed: {e}")
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
            self.log(f"Email interpretation failed: {e}")
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
        self.log(f"Gmail search started (scope={scope}, result={result_type}).")
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

    def _run_morning_routine(self, user_text: str) -> str:
        """Good morning: lights on at brightness 35, color to #FFB266 + brief weather + greeting."""
        stop_auto_lighting_sync(log_fn=self.log)
        # Lights: warm, gentle morning color on both systems.
        try:
            # Govee: warm white, 35% brightness.
            set_lights_style(state="on", color_hex=None, color_temp_k=2700, brightness=35)
        except LightsClientError as e:
            self.log(f"Govee morning routine failed: {e}")
        try:
            # Nanoleaf: warm orange, 35% brightness.
            nanoleaf.turn_on()
            # Approximate warm orange.
            nanoleaf.set_color_rgb(255, 180, 120)
            nanoleaf.set_brightness(35)
        except Exception as e:
            self.log(f"Nanoleaf morning routine failed: {e}")

        # Weather briefing if available.
        extra = ""
        try:
            weather = get_current_weather_summary()
            extra = (
                "System note: The app has already prepared the morning environment."
                " Briefly greet Andrew with 'Good morning, Andrew.' and naturally mention"
                f" the current weather: {weather}. Do NOT mention the lights at all, and"
                " do NOT repeat this note verbatim.\n\n"
            )
        except WeatherClientError as e:
            self.log(f"Weather fetch failed in morning routine: {e}")
            extra = (
                "System note: The app has already prepared the morning environment."
                " Briefly greet Andrew with 'Good morning, Andrew.' You do not know"
                " the current weather. Do NOT mention the lights, and do NOT repeat"
                " this note verbatim.\n\n"
            )
        self.log("Morning routine executed.")
        return self._call_model(user_text, None, extra_note=extra)

    def _run_night_routine(self, user_text: str) -> str:
        """Good night: turn lights off + goodnight message."""
        stop_auto_lighting_sync(log_fn=self.log)
        try:
            toggle_all_lights("off")
        except LightsClientError as e:
            self.log(f"Govee night routine failed: {e}")
        try:
            nanoleaf.turn_off()
        except Exception as e:
            self.log(f"Nanoleaf night routine failed: {e}")
        extra = (
            "System note: All lights have been turned off for the night."
            " Briefly say 'Goodnight, Andrew.' and nothing more.\n\n"
        )
        self.log("Night routine executed.")
        return self._call_model(user_text, None, extra_note=extra)

    def handle_message(self, user_text: str, user_name: str | None = None) -> str:
        """
        Route the message, run any tool, then call the model once. Return the assistant reply.
        """
        # Hard-wired routines for wake/sleep phrases so they are stable and fast.
        t_raw = (user_text or "").lower()
        normalized_name = (user_name or "").strip()
        self._active_user_name = normalized_name or "Andrew"

        # Optional micro-profiling: send "profile <message>" to print stage timings.
        profile_prefix = "profile "
        self._profile_active = bool(
            os.environ.get("GALADRIAL_PROFILE", "").lower() in ("1", "true", "yes")
            or t_raw.startswith(profile_prefix)
        )
        if t_raw.startswith(profile_prefix):
            user_text = (user_text or "")[len(profile_prefix) :].strip()
            t_raw = (user_text or "").lower()

        # Normalize common device spelling variants early (e.g. "nano leaf").
        user_text = re.sub(r"\bnano[\s-]*leaf\b", "nanoleaf", user_text or "", flags=re.IGNORECASE)
        t_raw = (user_text or "").lower()

        total_t0 = time.perf_counter() if self._profile_active else None

        if "good morning" in t_raw:
            return self._run_morning_routine(user_text)
        if "good night" in t_raw:
            return self._run_night_routine(user_text)

        if self._is_help_request(user_text):
            tool_lines = self._load_help_tools()
            if not tool_lines:
                return self._call_model(
                    user_text,
                    None,
                    extra_note=(
                        "System note: The app could not load tools.txt. "
                        "Give a short help reply that mentions core abilities: lights control, weather updates, and general questions.\n\n"
                    ),
                    compact_system=True,
                )
            tools_blob = "\n".join(f"- {item}" for item in tool_lines)
            return self._call_model(
                user_text,
                None,
                extra_note=(
                    "System note: The user asked for help. Give a concise natural-language summary of capabilities based on this list.\n"
                    "Do not mention files or internals.\n\n"
                    f"Capabilities list:\n{tools_blob}\n\n"
                ),
                compact_system=True,
            )

        # Deterministic weather handling: never guess if the API fails.
        if self._is_weather_forecast_query(user_text):
            try:
                return get_day_weather_forecast_summary()
            except WeatherClientError:
                return "I can't fetch the daily weather forecast right now. Please try again shortly."

        if self._is_weather_query(user_text):
            try:
                weather = get_current_weather_summary()
                return f"Right now it is {weather}."
            except WeatherClientError:
                return "I can't fetch the current weather right now. Please try again shortly."

        # Deterministic brightness status handling (read from Govee).
        if self._is_brightness_status_query(user_text):
            try:
                state = get_lights_state()
                st = state.get("state", "unknown")
                b = state.get("brightness")
                if isinstance(b, int):
                    return f"Govee is currently at {b}% brightness (and is {st})."
                return f"Govee is currently {st}, but I can't read brightness right now."
            except LightsClientError:
                return "I can't fetch the current brightness right now. Please try again shortly."

        if self._is_lights_set_to_query(user_text):
            try:
                state = get_lights_state()
                st = state.get("state", "unknown")
                b = state.get("brightness")
                mode = state.get("mode")
                color_hex = state.get("color_hex")
                color_temp_k = state.get("color_temp_k")

                parts: list[str] = []
                parts.append(f"Govee is currently {st}")
                if mode:
                    if isinstance(mode, str) and "auto" in mode.lower():
                        parts.append("in auto mode")
                    else:
                        parts.append(f"in {mode} mode")
                if isinstance(b, int):
                    parts.append(f"at {b}% brightness")

                # Color/temperature: only report what we can read.
                if isinstance(color_temp_k, int):
                    if color_temp_k < 3000:
                        parts.append(f"with warm white (about {color_temp_k}K)")
                    elif color_temp_k > 4200:
                        parts.append(f"with cool white (about {color_temp_k}K)")
                    else:
                        parts.append(f"with neutral white (about {color_temp_k}K)")
                elif isinstance(color_hex, str) and color_hex.strip().upper() == "#FFFFFF":
                    parts.append("with white")
                elif isinstance(color_hex, str) and color_hex.strip().startswith("#"):
                    # We don't try to map every hex to a friendly name; just show it.
                    parts.append(f"with color {color_hex.strip().upper()}")

                if len(parts) == 1:
                    return "I can read whether the Govee lights are on or off, but I can't read color or brightness right now."
                return " ".join(parts) + "."
            except LightsClientError:
                return "I can't fetch the current light settings right now. Please try again shortly."

        # Global alias expansion: if the whole message matches a remembered phrase, expand it
        effective_text = resolve_alias(user_text or "") or user_text

        # If the user explicitly asks us to remember something, store it deterministically
        # before any lighting routing (memory requests often contain light-related words).
        parsed_mapping = self._try_parse_remember_mapping(user_text)
        if parsed_mapping:
            key, value = parsed_mapping
            remember_alias(key, value)
            self.log("Memory: stored alias mapping (parsed).")
            return self._call_model(
                user_text,
                None,
                extra_note=f"System note: The app has just remembered that '{key}' means '{value}'. You can acknowledge that briefly.\n\n",
            )

        # Deterministic: auto mode enables Govee auto and keeps Nanoleaf synced every minute.
        t_lower = (effective_text or "").lower()
        if (
            ("auto" in t_lower or "automatic" in t_lower)
            and ("govee" in t_lower or "govee lights" in t_lower or "lights" in t_lower or "light" in t_lower)
            and "nanoleaf" not in t_lower
        ):
            try:
                start_auto_lighting_sync(log_fn=self.log)
            except Exception as e:
                self.log(f"Govee auto failed: {e}")
                return self._call_model(
                    effective_text,
                    None,
                    extra_note="System note: Setting Govee to auto mode failed. Tell the user it failed and they can try again.\n\n",
                )
            return self._call_model(
                effective_text,
                None,
                extra_note="System note: The app set Govee to auto mode and matched Nanoleaf to the same temperature and brightness profile. Nanoleaf will keep updating every minute while auto mode is active.\n\n",
            )

        # Deterministic: brightness up based on current lights settings.
        if self._is_too_dark_to_get_bright(effective_text):
            stop_auto_lighting_sync(log_fn=self.log)
            # Govee: try to read current brightness and bump it.
            target_brightness = None
            try:
                state = get_lights_state()
                cur = state.get("brightness")
                if isinstance(cur, int):
                    target_brightness = min(100, max(0, cur + 25))
            except LightsClientError:
                target_brightness = None

            if target_brightness is None:
                # Fallback when we cannot read current brightness.
                target_brightness = 100

            try:
                # Brightness-only: omit color/temperature so current color is preserved by the bulb.
                set_lights_style(state="on", brightness=target_brightness, color_hex=None, color_temp_k=None)
            except LightsClientError as e:
                self.log(f"Govee brightness up failed: {e}")

            try:
                nanoleaf.turn_on()
                nanoleaf.set_brightness(target_brightness)
            except Exception as e:
                self.log(f"Nanoleaf brightness up failed: {e}")

            return self._call_model(
                user_text,
                None,
                extra_note=f"System note: The app increased brightness to {target_brightness}%. Preserve the current lighting color/theme; do not change scenes.\n\n",
                compact_system=True,
            )

        # New structured-plan pipeline for lighting requests (semantic parse -> deterministic executor).
        # Keep the old tool-choice router for non-lighting (gmail/plex/etc.) and for memory.remember.
        if self._is_lighting_related(effective_text) and not self._is_memory_request(effective_text):
            scenes = nanoleaf.get_scene_list()
            plan = self._parse_lighting_plan(effective_text, scenes)
            if plan:
                cleaned = self._validate_lighting_plan(plan, effective_text, scenes)
                note = self._execute_lighting_plan(cleaned)
                self.log("Lighting plan executed.")

                # For lighting, we do NOT want the model to narrate logs like "Lights set: ...".
                # We rely on side effects only, and optionally give it weather for "good morning".
                extra = ""
                if "good morning" in (user_text or "").lower():
                    try:
                        weather = get_current_weather_summary()
                        extra = (
                            "System note: The app has already set a warm morning lighting scene."
                            f" Briefly greet Andrew with 'Good morning, Andrew.' and naturally mention"
                            f" the current weather: {weather}. Do NOT list internal details like"
                            " 'Lights set:' or repeat this note verbatim.\n\n"
                        )
                    except WeatherClientError as e:
                        self.log(f"Weather fetch failed: {e}")
                        extra = (
                            "System note: The app has already set a warm morning lighting scene."
                            " Briefly greet Andrew with 'Good morning, Andrew.' You do not know"
                            " the current weather.\n\n"
                        )

                return self._call_model(
                    effective_text,
                    None,
                    extra_note=extra,
                )

        # For non-lighting, non-weather, non-memory requests that are clearly just chat,
        # skip the tool-choice router LLM call (major latency win).
        if self._should_skip_tool_router(effective_text):
            self.last_route = {"action": "none", "params": {}}
            self.last_user_message = effective_text
            if self._profile_active and total_t0 is not None:
                self.log(f"PROFILE: router_model_call=0.00s (skipped router)")
            reply = self._call_model(effective_text, None, compact_system=True)
            if self._profile_active and total_t0 is not None:
                self.log(f"PROFILE: total={time.perf_counter() - total_t0:.2f}s")
            return reply

        route = self.route(effective_text)
        action = (route.get("action") or "none").lower()
        params = route.get("params") or {}
        self.last_route = route
        self.last_user_message = effective_text

        if action == "memory.remember":
            key = str(params.get("key") or "").strip()
            value = str(params.get("value") or "").strip()
            if key and value:
                remember_alias(key, value)
                self.log("Memory: stored alias mapping.")
                return self._call_model(
                    user_text,
                    None,
                    extra_note=f"System note: The app has just remembered that '{key}' means '{value}'. You can acknowledge that briefly.\n\n",
                )
            # Model didn't give usable params; just answer normally
            return self._call_model(
                user_text,
                None,
                extra_note="System note: The app could not save that memory because key or value was missing. You may ask the user to restate it.\n\n",
            )

        lighting_reply = try_handle_lighting_action(self, action, params, user_text, effective_text)
        if lighting_reply is not None:
            return lighting_reply

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

        reply = self._call_model(user_text, None)
        if self._profile_active and total_t0 is not None:
            self.log(f"PROFILE: total={time.perf_counter() - total_t0:.2f}s")
        return reply


def handle_message(
    user_text: str,
    log_fn: Optional[Callable[[str], None]] = None,
    user_name: str | None = None,
) -> str:
    """
    One-shot: create engine, handle message, return reply.
    Use this from FastAPI or scripts. For multi-turn with follow-ups, use AssistantEngine() and call handle_message on it.
    """
    engine = AssistantEngine(log_fn=log_fn)
    return engine.handle_message(user_text, user_name=user_name)


if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Are the lights on?"
    print("User:", msg)
    reply = handle_message(msg, log_fn=print)
    print("Reply:", reply)
