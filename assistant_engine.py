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
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Optional

from llm import ask_lmstudio
from lighting.lights_client import (
    get_lights_state,
    toggle_all_lights,
    LightsClientError,
    set_lights_auto,
    set_lights_style,
)
from lighting.nanoleaf import nanoleaf
from misc_tools.user_memory import remember_alias, resolve_alias
from misc_tools.weather_client import (
    get_current_weather_summary,
    get_day_weather_forecast_summary,
    WeatherClientError,
)

# Plex sync (same as chat_gui; override with PLEX_SYNC_DIR env if needed)
PLEX_SYNC_DIR = os.environ.get("PLEX_SYNC_DIR", r"H:\Coding\Python Projects\plex_sync")
PLEX_SYNC_PY = os.path.join(PLEX_SYNC_DIR, ".venv", "Scripts", "python.exe")
PLEX_SYNC_MAIN = os.path.join(PLEX_SYNC_DIR, "main.py")
TOOLS_HELP_FILE = os.path.join(os.path.dirname(__file__), "tools.txt")

from assistant_engine_tools import TOOLS, VALID_ACTIONS, _format_tools_for_prompt
from lighting.assistant_engine_lighting import (
    get_last_nanoleaf_flow,
    infer_flow_colors_hex,
    infer_flow_speed,
    persist_last_nanoleaf_flow,
    try_handle_lighting_action,
)
from lighting.auto_lighting_sync import start_auto_lighting_sync, stop_auto_lighting_sync
from music.play_resolver import (
    format_play_resolution_reply,
    looks_like_play_music_request,
    resolve_play_to_video_id,
)
from music.spotify_resolver import (
    format_spotify_resolution_reply,
    looks_like_spotify_pause_request,
    looks_like_spotify_play_request,
    pause_spotify_playback,
    resolve_spotify_play,
)


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
        self.last_timing_ms: dict[str, float] | None = None
        self._router_llm_ms_sum: float = 0.0
        self._reply_llm_ms_sum: float = 0.0

    @contextmanager
    def _timing_scope(self):
        """Wall-clock and per-LLM-call timings for /chat profiling."""
        wall_start = time.perf_counter()
        self._router_llm_ms_sum = 0.0
        self._reply_llm_ms_sum = 0.0
        self.last_timing_ms = None
        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - wall_start) * 1000
            r = self._router_llm_ms_sum
            p = self._reply_llm_ms_sum
            self.last_timing_ms = {
                "total_ms": round(wall_ms, 1),
                "router_llm_ms": round(r, 1),
                "reply_llm_ms": round(p, 1),
                "other_ms": round(max(0.0, wall_ms - r - p), 1),
            }

    def _is_wake_sleep_phrase(self, text: str) -> bool:
        """Match good morning/night variants like 'goodnight' and 'good-night'."""
        t = (text or "").lower().strip()
        if not t:
            return False
        return bool(re.search(r"\bgood[\s-]*morning\b", t) or re.search(r"\bgood[\s-]*night\b", t))

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
                # Phrases like "create an animation with rainbow colors" often omit "lights".
                "animation",
                "animate",
                "pulse",
                "rainbow",
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

        # Explicit day references.
        if any(k in t for k in ("tomorrow", "day after tomorrow", "today")):
            return True
        # Explicit calendar date like "March 30th".
        if re.search(
            r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|"
            r"august|aug|september|sep|sept|october|oct|november|nov|december|dec)\s+"
            r"\d{1,2}(?:st|nd|rd|th)?\b",
            t,
        ):
            return True
        if any(
            re.search(rf"\b{rx}\b", t)
            for rx in (
                "monday|mon",
                "tuesday|tue",
                "wednesday|wed",
                "thursday|thu",
                "friday|fri",
                "saturday|sat",
                "sunday|sun",
            )
        ):
            return True

        # Weather forecast signals.
        if any(k in t for k in ("forecast", "today", "for the day", "precipitation", "precip")):
            return True
        if any(k in t for k in ("high", "peak", "tonight")):
            return True
        # "rain" should be a standalone word to avoid matching "rainbow".
        if re.search(r"\brain\b", t):
            return True
        return False

    def _weather_forecast_day_offset(self, text: str) -> int:
        """
        Map user text to a day offset relative to "today" (0=today, 1=tomorrow, ...).

        Examples handled:
        - "tomorrow", "day after tomorrow"
        - weekday names like "Wednesday" (next occurrence; can be today if asked on Wednesday)
        - "next Wednesday" (forces a +7 jump when the weekday matches today)
        """
        t = (text or "").lower()

        if "day after tomorrow" in t:
            return 2
        if "tomorrow" in t:
            return 1
        if "today" in t:
            return 0

        weekday_map = {
            0: ("monday", "mon"),
            1: ("tuesday", "tue"),
            2: ("wednesday", "wed"),
            3: ("thursday", "thu"),
            4: ("friday", "fri"),
            5: ("saturday", "sat"),
            6: ("sunday", "sun"),
        }
        token = None
        for offset, aliases in weekday_map.items():
            for a in aliases:
                if re.search(rf"\b{a}\b", t):
                    token = offset
                    break
            if token is not None:
                break

        if token is None:
            # Try parsing explicit calendar dates like "March 30th".
            month_map = {
                "january": 1,
                "jan": 1,
                "february": 2,
                "feb": 2,
                "march": 3,
                "mar": 3,
                "april": 4,
                "apr": 4,
                "may": 5,
                "june": 6,
                "jun": 6,
                "july": 7,
                "jul": 7,
                "august": 8,
                "aug": 8,
                "september": 9,
                "sep": 9,
                "sept": 9,
                "october": 10,
                "oct": 10,
                "november": 11,
                "nov": 11,
                "december": 12,
                "dec": 12,
            }

            m = re.search(
                r"\b(?P<month>january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|"
                r"august|aug|september|sep|sept|october|oct|november|nov|december|dec)\s+"
                r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(?P<year>\d{4}))?\b",
                t,
            )
            if m:
                month_s = (m.group("month") or "").lower()
                day_s = m.group("day") or ""
                year_s = m.group("year")
                try:
                    month = int(month_map.get(month_s) or 0)
                    day = int(day_s)
                    now_date = datetime.now().date()
                    year = int(year_s) if year_s else now_date.year
                    target = datetime(year, month, day).date()
                    if not year_s and target < now_date:
                        # If no year is specified and the date already passed this year,
                        # assume the next occurrence.
                        target = datetime(year + 1, month, day).date()
                    offset = (target - now_date).days
                    return max(0, int(offset))
                except Exception:
                    return 0

            return 0

        now = datetime.now()
        today_wd = now.weekday()  # Monday=0
        target_wd = token
        delta = (target_wd - today_wd) % 7

        if delta == 0 and "next" in t:
            delta = 7

        # Clamp to our forecast helper's supported range (weekday names only).
        return max(0, min(7, int(delta)))

    def _weather_forecast_target_hour(self, text: str) -> int | None:
        """
        Parse optional time-of-day from weather query.
        Supports: noon, midnight, 1pm, 1 pm, 1 p. m., 1:30pm, 13:00.
        Returns hour in 24h form (0..23), or None if no time mentioned.
        """
        t = (text or "").lower()
        if "noon" in t:
            return 12
        if "midnight" in t:
            return 0

        # Normalize dotted/spaced meridiem forms:
        # "2 p. m." / "2 p.m." / "2 p m" -> "2 pm"
        t = re.sub(r"\b([ap])\s*\.?\s*m\.?\b", r"\1m", t)

        # 1pm / 1 pm / 1:30pm / 1:30 pm
        m12 = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b", t)
        if m12:
            hour = int(m12.group(1))
            ampm = m12.group(3)
            hour = max(1, min(12, hour))
            if ampm == "am":
                return 0 if hour == 12 else hour
            return 12 if hour == 12 else hour + 12

        # 24h like 13:00
        m24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
        if m24:
            return int(m24.group(1))

        return None

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
        If the message is clearly just general chat (no lights/plex/memory/weather actions),
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
- For type \"animation\" you MUST set \"colors_hex\" to an array of at least 2 hex strings. Examples: rainbow -> 6+ distinct hues (#FF0000, #FF8800, #FFFF00, #00FF00, #0088FF, #8800FF); red and blue -> [\"#FF0000\", \"#0000FF\"]. Never leave colors_hex empty for animation.

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
                if cols is None and isinstance(a.get("colors"), list):
                    cols = a.get("colors")
                if isinstance(cols, list):
                    out_cols = []
                    for c in cols:
                        s = str(c).strip()
                        if s.startswith("#") and len(s) == 7:
                            out_cols.append(s)
                        elif len(s) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in s):
                            out_cols.append("#" + s.upper())
                    out["colors_hex"] = out_cols[:6]
                else:
                    out["colors_hex"] = []
                if len(out["colors_hex"]) < 2:
                    out["colors_hex"] = infer_flow_colors_hex(user_text)[:6]
                sp = a.get("speed")
                try:
                    out["speed"] = (
                        max(0.5, min(5.0, float(sp)))
                        if sp is not None
                        else infer_flow_speed(user_text)
                    )
                except Exception:
                    out["speed"] = infer_flow_speed(user_text)
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
        # Parsed lighting intent: always cancel Nanoleaf time-sync and Govee auto, even if validation
        # dropped all actions (otherwise auto mode keeps winning over a custom scene).
        stop_auto_lighting_sync(log_fn=self.log)
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
                            ok = nanoleaf.create_flow_effect(rgb, float(speed))
                            if ok:
                                notes.append("Nanoleaf flow animation applied.")
                                persist_last_nanoleaf_flow(self._active_user_name, cols, float(speed))
                            else:
                                self.log("Nanoleaf create_flow_effect returned False.")
                                notes.append("Nanoleaf flow animation failed (API rejected effect).")
                        else:
                            notes.append("Nanoleaf animation skipped (not enough colors after validation).")
                    except Exception as e:
                        self.log(f"Nanoleaf animation failed: {e}")
                        notes.append("Nanoleaf animation failed.")
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
        """
        Re-apply the last custom flow with a new speed when the user asks faster/slower.
        Uses last_route in-process or persisted state (each HTTP request uses a new engine).
        Nanoleaf speed param: higher value = faster motion (see create_flow_effect).
        """
        prev_params: dict | None = None
        if self.last_route and self.last_route.get("action") == "nanoleaf.create_animation":
            pp = self.last_route.get("params") or {}
            cols = pp.get("colors")
            if isinstance(cols, list) and len(cols) >= 2:
                prev_params = dict(pp)
        if prev_params is None:
            stored = get_last_nanoleaf_flow(getattr(self, "_active_user_name", None))
            if stored and isinstance(stored.get("colors"), list) and len(stored["colors"]) >= 2:
                prev_params = {
                    "colors": list(stored["colors"]),
                    "speed": float(stored.get("speed", 1.0)),
                }
        if not prev_params:
            return None
        colors = prev_params.get("colors")
        if not isinstance(colors, list) or len(colors) < 2:
            return None
        t = user_text.lower().strip()
        try:
            current = float(prev_params.get("speed", 1.0))
        except (TypeError, ValueError):
            current = 1.0
        current = max(0.5, min(5.0, current))
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
                "pulse faster",
                "animate faster",
                "cycle faster",
                "quicker",
                "more quickly",
            )
        ):
            if "super fast" in t or "really fast" in t or "maximum speed" in t or "max speed" in t:
                new_speed = 5.0
            elif "a little" in t:
                new_speed = min(5.0, round(current * 1.2, 1))
            else:
                new_speed = min(5.0, round(current * 1.45, 1))
            return {
                "action": "nanoleaf.create_animation",
                "params": {
                    "animation_type": "flow",
                    "colors": colors,
                    "speed": new_speed,
                },
            }
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
                "pulse slower",
            )
        ):
            if "a little" in t:
                new_speed = max(0.5, round(current * 0.85, 1))
            else:
                new_speed = max(0.5, round(current * 0.65, 1))
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

    def _route_flow_animation(self, user_text: str) -> dict | None:
        """
        If the user clearly wants a new flowing/custom animation, return nanoleaf.create_animation
        with colors from infer_flow_colors_hex — no lighting-plan LLM or tool router required.
        """
        if self._is_memory_request(user_text):
            return None
        t = (user_text or "").lower().strip()
        if not (
            ("animation" in t or "animate" in t)
            and any(k in t for k in ("create", "custom", "make", "new", "build", "add"))
        ) and not ("rainbow" in t and ("animation" in t or "animate" in t or "color" in t)):
            return None
        colors = infer_flow_colors_hex(user_text)
        spd = infer_flow_speed(user_text)
        self.log(f"Heuristic flow animation; colors={colors} speed={spd}")
        return {
            "action": "nanoleaf.create_animation",
            "params": {
                "animation_type": "flow",
                "colors": colors,
                "speed": spd,
            },
        }

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
- "make it faster" / "pulse faster" / "speed it up" / "slower" (after a custom Nanoleaf flow was applied) → nanoleaf.create_animation again with the same colors and new speed (the app may also handle this without you).
- "turn the nanoleaf off" / "nanoleaf on" / "turn off the nanoleaf" → nanoleaf.set_state with state "off" or "on" (ONLY Nanoleaf; do NOT change Govee). Do NOT use nanoleaf.custom for this.
- "make the nanoleaf lights dimmer" / "nanoleaf brighter" / "set nanoleaf to 50%" → nanoleaf.set_brightness (ONLY Nanoleaf brightness; do NOT change scene or Govee). Do NOT use lights.set_scene for this.
- "turn lights on" / "lights off" (all lights) → lights.set_state with state "on" or "off"
- "are the lights on?" → lights.get_state
- "run Plex sync" → plex_sync.run
- "remember that writing mode means dim orange lights" → memory.remember (key: "writing mode", value: "dim orange lights")

If the user said "again" or "same thing", repeat the previous action. Do not choose "none" when the user is asking you to change the lights or set a mood—use lights.set_scene or nanoleaf.set_scene as appropriate. If they ask to "create a new animation" or "create an animation for the lights", always use nanoleaf.create_animation with colors (and optional speed)—never respond with a list of settings for them to enter in an app.

Previous user message: "{prev_msg}"
Previous action: {prev_action}

User message: "{user_text}"

Respond with JSON only:
{{"action": "<tool name from the list above>", "params": {{...}}}}
"""
        try:
            t0 = time.perf_counter()
            response = ask_lmstudio(tool_choice_prompt)
            self._router_llm_ms_sum += (time.perf_counter() - t0) * 1000
            if self._profile_active:
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
                " (for example, Plex sync, weather). Those System messages are the"
                " source of truth about my real data.\n"
                "- VERY IMPORTANT: Do NOT repeat System notes verbatim or list them like logs"
                " (e.g. 'Lights set:', 'Weather:', 'Quick news summary:'). Instead"
                " briefly and naturally incorporate only the important parts"
                " into your reply, or skip details I didn't explicitly ask for.\n\n"
                "CRITICAL SAFETY RULES:\n"
                "- Do NOT invent or guess specific facts.\n"
                "- If you are not given an explicit System note or tool result that contains those"
                " facts, you must say that you don't know instead of making something up.\n"
                "- You may still answer general questions with your own knowledge, but never fabricate"
                " concrete details about my life or accounts.\n\n"
            "- Weather safety: If the user asks about weather (current or forecast), you may only state"
            " it when you were given a System note or tool result that contains the weather."
            " Otherwise, say you don't know.\n\n"
            )
        action_note = ""
        if light_action in ("on", "off", "auto"):
            action_note = f"System note: The app has just set the lights to '{light_action}'.\n\n"
        if extra_note:
            action_note = action_note + extra_note
        full_prompt = f"{system_preamble}{action_note}User: {prompt}"
        try:
            t0 = time.perf_counter()
            response = ask_lmstudio(full_prompt)
            self._reply_llm_ms_sum += (time.perf_counter() - t0) * 1000
            if self._profile_active:
                self.log(f"PROFILE: assistant_model_call={time.perf_counter() - t0:.2f}s")
            return (response.get("output") or [{}])[0].get("content", "").strip() or "No response."
        except Exception as e:
            self.log(f"Model error: {e}")
            return f"Error calling model: {e}"

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
        with self._timing_scope():
            return self._handle_message_impl(user_text, user_name)

    def _handle_message_impl(self, user_text: str, user_name: str | None = None) -> str:
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

        if self._is_wake_sleep_phrase(t_raw) and re.search(r"\bgood[\s-]*morning\b", t_raw):
            return self._run_morning_routine(user_text)
        if self._is_wake_sleep_phrase(t_raw) and re.search(r"\bgood[\s-]*night\b", t_raw):
            return self._run_night_routine(user_text)

        # Deterministic Spotify pause/stop.
        if looks_like_spotify_pause_request(user_text):
            ok, detail, _dev = pause_spotify_playback()
            if ok:
                return "Paused Spotify playback."
            return f"I couldn't pause Spotify playback: {detail or 'unknown error'}."

        # Spotify before generic "Play …" (YouTube).
        # If Spotify playback can't start and the user did NOT explicitly mention Spotify,
        # fall back to the existing YouTube pipeline.
        explicit_spotify = "spotify" in (user_text or "").lower()
        if looks_like_spotify_play_request(user_text):
            sp_res = resolve_spotify_play(user_text)
            if sp_res.ok and (sp_res.playback_started or explicit_spotify):
                return format_spotify_resolution_reply(sp_res)
            # Soft fallback only when the user didn't explicitly ask for Spotify.
            if not explicit_spotify and looks_like_play_music_request(user_text):
                res = resolve_play_to_video_id(user_text)
                return format_play_resolution_reply(res)
            return format_spotify_resolution_reply(sp_res)

        # YouTube "Play …" → search → video_id (pass to Pi / player separately).
        if looks_like_play_music_request(user_text):
            res = resolve_play_to_video_id(user_text)
            return format_play_resolution_reply(res)

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
                day_offset = self._weather_forecast_day_offset(user_text)
                target_hour = self._weather_forecast_target_hour(user_text)
                return get_day_weather_forecast_summary(
                    day_offset=day_offset,
                    target_hour_24=target_hour,
                )
            except WeatherClientError as e:
                emsg = (str(e) or "").strip()
                if emsg:
                    # For range errors, we want the user-facing message.
                    if "forecast up to" in emsg or "days away" in emsg:
                        return emsg
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

        # Deterministic flow animations: skip lighting-plan + router LLM when intent is unambiguous.
        flow_route = self._route_flow_animation(effective_text)
        if flow_route is not None:
            self.last_route = flow_route
            self.last_user_message = effective_text
            lighting_reply = try_handle_lighting_action(
                self,
                flow_route["action"],
                flow_route["params"],
                user_text,
                effective_text,
            )
            if lighting_reply is not None:
                return lighting_reply

        # New structured-plan pipeline for lighting requests (semantic parse -> deterministic executor).
        # Keep the old tool-choice router for non-lighting (plex/memory/etc.).
        if self._is_lighting_related(effective_text) and not self._is_memory_request(effective_text):
            scenes = nanoleaf.get_scene_list()
            plan = self._parse_lighting_plan(effective_text, scenes)
            if plan:
                cleaned = self._validate_lighting_plan(plan, effective_text, scenes)
                note = self._execute_lighting_plan(cleaned)
                self.log("Lighting plan executed.")

                # Tell the model what actually happened (prevents empty affirmations when hardware did nothing).
                extra = ""
                if note and note.strip():
                    if note.strip() == "No lighting changes applied.":
                        extra = (
                            "System note: No lighting actions were applied (plan had no executable steps). "
                            "Do NOT claim you changed the lights; briefly say it did not work or ask what look they want.\n\n"
                        )
                    else:
                        extra = f"System note: Lighting result: {note}\n\n"

                # Optionally add weather for "good morning" on top of the lighting note.
                if "good morning" in (user_text or "").lower():
                    try:
                        weather = get_current_weather_summary()
                        gm = (
                            "The app has already set a warm morning lighting scene."
                            f" Briefly greet Andrew with 'Good morning, Andrew.' and naturally mention"
                            f" the current weather: {weather}. Do NOT list internal details like"
                            " 'Lights set:' or repeat this note verbatim.\n\n"
                        )
                        extra = (extra or "") + "System note: " + gm
                    except WeatherClientError as e:
                        self.log(f"Weather fetch failed: {e}")
                        gm = (
                            "The app has already set a warm morning lighting scene."
                            " Briefly greet Andrew with 'Good morning, Andrew.' You do not know"
                            " the current weather.\n\n"
                        )
                        extra = (extra or "") + "System note: " + gm

                return self._call_model(
                    effective_text,
                    None,
                    extra_note=extra or None,
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
    timing_out: dict | None = None,
) -> str:
    """
    One-shot: create engine, handle message, return reply.
    Use this from FastAPI or scripts. For multi-turn with follow-ups, use AssistantEngine() and call handle_message on it.
    If ``timing_out`` is a dict, it is filled with ``last_timing_ms`` keys (total_ms, router_llm_ms, …).
    """
    engine = AssistantEngine(log_fn=log_fn)
    reply = engine.handle_message(user_text, user_name=user_name)
    if timing_out is not None:
        timing_out.clear()
        if engine.last_timing_ms:
            timing_out.update(engine.last_timing_ms)
    return reply


if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Are the lights on?"
    print("User:", msg)
    reply = handle_message(msg, log_fn=print)
    print("Reply:", reply)
