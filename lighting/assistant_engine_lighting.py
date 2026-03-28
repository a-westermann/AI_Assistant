"""
Lighting action handlers extracted from assistant_engine.py.
"""

from __future__ import annotations

import colorsys
import json
import re
from collections.abc import Callable
from typing import Any

from llm import ask_lmstudio

from .auto_lighting_sync import start_auto_lighting_sync, stop_auto_lighting_sync
from .lights_client import (
    get_lights_state,
    toggle_all_lights,
    LightsClientError,
    set_lights_auto,
    set_lights_style,
)
from .nanoleaf import nanoleaf


def _is_lighting_action(action: str) -> bool:
    a = (action or "").strip().lower()
    return a.startswith("lights.") or a.startswith("nanoleaf.")


def _should_cancel_auto_sync_for_action(action: str, params: dict[str, Any]) -> bool:
    """
    Manual lighting changes should cancel auto sync.
    Keep auto running only for explicit "lights.set_state" with state=auto,
    and for read-only status checks.
    """
    a = (action or "").strip().lower()
    if a in ("lights.get_state",):
        return False
    if a == "lights.set_state":
        state = str((params or {}).get("state") or "").strip().lower()
        return state != "auto"
    return True


def pick_govee_style(description: str, log_fn) -> dict | None:
    """Ask LLM for color/temperature/brightness to fit a mood."""
    prompt = (
        f'The user wants their room lights to feel "{description}".\n\n'
        "Suggest settings for a smart bulb: color (hex #RRGGBB), color temperature in Kelvin (2200-3000 warm, 4000-6500 cool), and brightness 0-100. "
        "You MUST always include brightness. If they say dimmer, dim, or dark use brightness 25-45; if they say brighter or bright use 75-100; otherwise 50-70. "
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
        log_fn(f"Govee style pick failed: {e}")
        return None


def pick_nanoleaf_style(description: str, log_fn) -> dict | None:
    """Ask LLM for Nanoleaf color and brightness to fit a mood."""
    prompt = (
        f'The user said: "{description}".\n\n'
        "Reply with JSON: color_hex (hex #RRGGBB) and brightness (0-100). "
        "If they name a color (green, purple, blue, red, yellow, etc.) you MUST use that color-e.g. green -> #00FF00 or #228B22, purple -> #800080, blue -> #0066FF. "
        "Only use white (#FFFFFF) when they mention no color and only brightness (dimmer, brighter, etc.). "
        "Brightness: max bright / bright as possible -> 100; dimmer / dim -> 25-40; brighter -> 75-90; otherwise 60-80. "
        "Reply with JSON only, no explanation:\n"
        '{"color_hex": "#RRGGBB", "brightness": 70}'
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
        log_fn(f"Nanoleaf style pick failed: {e}")
        return None


def pick_nanoleaf_scene(description: str, scene_list: list[str], log_fn) -> str | None:
    """Pick the best predefined Nanoleaf scene name for a mood."""
    if not scene_list or not description.strip():
        return None
    scene_list_str = ", ".join(scene_list)
    prompt = (
        f"Available Nanoleaf scene names (pick exactly one):\n{scene_list_str}\n\n"
        f"The user said they want: \"{description.strip()}\"\n\n"
        "They may use words that are not exact scene names (e.g. 'sexy', 'chill', 'cozy', 'intimate'). "
        "Choose the scene that best fits the mood or feeling they are describing. "
        "For example, 'sexy' or 'intimate' are similar to Romantic; 'calm' might match Inner Peace or Snowfall. "
        "Reply with ONLY the single scene name from the list above-no explanation, no quotes, just the name."
    )
    try:
        response = ask_lmstudio(prompt)
        raw = (response.get("output") or [{}])[0].get("content", "").strip()
        if not raw:
            return None
        first_line = raw.split("\n")[0].strip().strip("\"'")
        first_line_lower = first_line.lower()
        for name in scene_list:
            if name.lower() == first_line_lower:
                return name
        for name in scene_list:
            if name.lower() in first_line_lower or first_line_lower in name.lower():
                return name
        return None
    except Exception as e:
        log_fn(f"Nanoleaf scene pick failed: {e}")
        return None


def infer_flow_speed(user_text: str) -> float:
    """
    Map natural language to Nanoleaf transition-style speed value.
    Lower values = faster motion (e.g. 0.1 is very fast), higher = slower.
    """
    t = (user_text or "").lower()
    if "really slow" in t or "very slow" in t or "super slow" in t:
        return 4.5
    if "really fast" in t or "super fast" in t or "very fast" in t:
        return 0.2
    if "maximum speed" in t or "max speed" in t or "as fast as" in t:
        return 0.1
    if any(w in t for w in ("slow", "slowly", "gentle", "calm", "leisurely", "subtle")):
        return 3.0
    if any(w in t for w in ("fast", "quick", "quickly", "snappy", "rapid", "speedy")):
        return 0.7
    return 1.0


def pick_flow_palette_for_reference(
    context: str, log_fn: Callable[[str], None]
) -> tuple[list[str], float | None] | None:
    """
    Use the LLM to map a game / movie / show / book / mood reference (no explicit color names required)
    to 2–6 hex colors for a Nanoleaf flowing animation. Optionally returns a speed hint (0.1 fast … 5 slow).
    """
    ctx = (context or "").strip()
    if len(ctx) < 12:
        return None
    prompt = (
        "The user wants a FLOWING / ANIMATED multi-color effect on smart light panels (Nanoleaf) that fits "
        "this situation, activity, or reference:\n\n"
        f'"{ctx}"\n\n'
        "Pick 2 to 6 distinct HTML hex colors (#RRGGBB) that evoke the look and mood—think iconic palette, "
        "lighting, atmosphere, or brand colors associated with that reference. "
        "Use saturated, panel-friendly colors (avoid many near-duplicates). "
        "Optionally set \"speed\": a number from 0.1 (fast, energetic) to 5.0 (slow, calm) for how quickly "
        "colors should transition—match the pacing of the reference (horror often slower, arcade/brighter often faster).\n"
        "Examples: survival horror game -> deep crimson, near-black burgundy, sickly yellow-green, cold gray-blue; "
        "underwater documentary -> deep blue, cyan, bioluminescent blue-green; "
        "cozy reading -> warm amber, soft orange, dim gold.\n"
        'Reply with JSON only, no markdown: {"colors_hex": ["#RRGGBB", "#RRGGBB"], "speed": 2.0}\n'
        "Include speed only when it clearly fits the mood; omit speed if unsure."
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
        arr = obj.get("colors_hex")
        if not isinstance(arr, list):
            return None
        out: list[str] = []
        for c in arr[:6]:
            s = str(c).strip()
            if s.startswith("#") and len(s) == 7:
                try:
                    int(s[1:], 16)
                    out.append(s.upper())
                except ValueError:
                    continue
            elif len(s) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in s):
                try:
                    int(s, 16)
                    out.append("#" + s.upper())
                except ValueError:
                    continue
        if len(out) < 2:
            return None
        theme_sp: float | None = None
        sp = obj.get("speed")
        if isinstance(sp, (int, float)) and not isinstance(sp, bool):
            theme_sp = max(0.1, min(5.0, float(sp)))
        elif isinstance(sp, str):
            try:
                theme_sp = max(0.1, min(5.0, float(sp.strip())))
            except ValueError:
                pass
        return out, theme_sp
    except Exception as e:
        log_fn(f"Flow palette from reference failed: {e}")
    return None


_LAST_NANOLEAF_FLOW_BY_USER: dict[str, dict[str, Any]] = {}


def _flow_state_user_key(user_name: str | None) -> str:
    n = (user_name or "").strip().lower()
    return n if n else "default"


def persist_last_nanoleaf_flow(user_name: str | None, colors_hex: list[str], speed: float) -> None:
    """Remember last custom flow so 'faster/slower' works across separate /chat requests (new engine each time)."""
    if len(colors_hex) < 2:
        return
    norm: list[str] = []
    for c in colors_hex[:6]:
        s = str(c).strip()
        if s.startswith("#") and len(s) == 7:
            norm.append(s.upper())
        elif len(s) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in s):
            norm.append("#" + s.upper())
    if len(norm) < 2:
        return
    sp = max(0.1, min(5.0, round(float(speed), 2)))
    _LAST_NANOLEAF_FLOW_BY_USER[_flow_state_user_key(user_name)] = {"colors": norm, "speed": sp}


def get_last_nanoleaf_flow(user_name: str | None) -> dict[str, Any] | None:
    """Snapshot {colors: [#RRGGBB, ...], speed} or None."""
    return _LAST_NANOLEAF_FLOW_BY_USER.get(_flow_state_user_key(user_name))


def clear_last_nanoleaf_flow(user_name: str | None = None) -> None:
    """
    Clear remembered custom flow state.
    - user_name=None: clear all users
    - user_name provided: clear only that user's flow snapshot
    """
    if user_name is None:
        _LAST_NANOLEAF_FLOW_BY_USER.clear()
        return
    _LAST_NANOLEAF_FLOW_BY_USER.pop(_flow_state_user_key(user_name), None)


# Color names -> hex for inferring Nanoleaf flow palettes from free text.
# Longer keys matched first so e.g. "light blue" wins over "blue" when we add phrases.
_FLOW_NAMED_HEX: dict[str, str] = {
    "light blue": "#87CEEB",
    "sky blue": "#87CEEB",
    "royal blue": "#4169E1",
    "navy": "#000080",
    "navy blue": "#000080",
    "baby blue": "#89CFF0",
    "turquoise": "#40E0D0",
    "aqua": "#00FFFF",
    "cyan": "#00BCD4",
    "teal": "#008080",
    "mint": "#98FF98",
    "forest": "#228B22",
    "forest green": "#228B22",
    "lime": "#32CD32",
    "olive": "#808000",
    "emerald": "#50C878",
    "chartreuse": "#7FFF00",
    "neon green": "#39FF14",
    "hot pink": "#FF69B4",
    "magenta": "#FF00FF",
    "fuchsia": "#FF00FF",
    "violet": "#8B00FF",
    "indigo": "#4B0082",
    "lavender": "#E6E6FA",
    "crimson": "#DC143C",
    "scarlet": "#FF2400",
    "maroon": "#800000",
    "burgundy": "#800020",
    "coral": "#FF7F50",
    "peach": "#FFCBA4",
    "salmon": "#FA8072",
    "tan": "#D2B48C",
    "beige": "#F5F5DC",
    "ivory": "#FFFFF0",
    "cream": "#FFFDD0",
    "silver": "#C0C0C0",
    "charcoal": "#36454F",
    "gray": "#808080",
    "grey": "#808080",
    "black": "#222222",
    "white": "#F5F5F5",
    "gold": "#FFD700",
    "amber": "#FFBF00",
    "bronze": "#CD7F32",
    "copper": "#B87333",
    "rose": "#FF007F",
    "red": "#FF0000",
    "orange": "#FF8800",
    "yellow": "#FFEB3B",
    "green": "#00C853",
    "blue": "#2196F3",
    "purple": "#9C27B0",
    "pink": "#FF69B4",
}


def _ordered_named_colors_from_text(t: str) -> list[str]:
    """
    Find color words in user text in left-to-right order; dedupe adjacent duplicates.
    Uses word boundaries so 'blue' does not match inside 'bluetooth'.
    """
    t = (t or "").lower()
    names = sorted(_FLOW_NAMED_HEX.keys(), key=len, reverse=True)
    matches: list[tuple[int, str]] = []
    for name in names:
        hx = _FLOW_NAMED_HEX[name]
        for m in re.finditer(rf"\b{re.escape(name)}\b", t):
            matches.append((m.start(), hx))
    matches.sort(key=lambda x: x[0])
    out: list[str] = []
    for _, hx in matches:
        if not out or out[-1] != hx:
            out.append(hx)
    return out[:6]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    s = h.strip()
    if s.startswith("#") and len(s) == 7:
        return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
    return 255, 255, 255


def _pair_single_color_flow(hex_color: str) -> list[str]:
    """If only one color was named, add a second stop by rotating hue (~45°) so flow still works."""
    r, g, b = _hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h2 = (h + 45.0 / 360.0) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h2, max(0.35, s), max(0.45, v))
    second = "#{:02X}{:02X}{:02X}".format(
        int(r2 * 255) & 255,
        int(g2 * 255) & 255,
        int(b2 * 255) & 255,
    )
    return [hex_color.upper() if hex_color.startswith("#") else "#" + hex_color, second]


def resolve_nanoleaf_flow_colors(
    user_text: str,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[str], float]:
    """
    Build flow palette and a speed value from user text.
    Uses infer_flow_speed for keyword-based pacing; when the thematic LLM runs, its optional speed
    overrides that baseline for the same message.
    """
    t = (user_text or "").lower()
    raw_ctx = (user_text or "").strip()
    base_speed = infer_flow_speed(user_text)

    if "rainbow" in t:
        return (
            ["#FF0000", "#FF8800", "#FFFF00", "#00FF00", "#0088FF", "#8800FF"],
            base_speed,
        )
    if re.search(r"\bsunset\b", t):
        return (["#FF4500", "#FF8C00", "#FFD700"], base_speed)
    if re.search(r"\bocean\b", t) or re.search(r"\bsea\b", t):
        return (["#001830", "#0066CC", "#00CED1"], base_speed)

    ordered = _ordered_named_colors_from_text(t)
    if len(ordered) >= 2:
        return ordered, base_speed
    if len(ordered) == 1:
        return _pair_single_color_flow(ordered[0]), base_speed

    if log_fn is not None and len(raw_ctx) >= 12:
        picked = pick_flow_palette_for_reference(raw_ctx, log_fn)
        if picked:
            themed_colors, theme_sp = picked
            log_fn(f"Infer flow: thematic palette from reference ({len(themed_colors)} colors).")
            spd = theme_sp if theme_sp is not None else base_speed
            return themed_colors, spd

    return ["#4A90D9", "#7EC8E3"], base_speed


def infer_flow_colors_hex(
    user_text: str,
    log_fn: Callable[[str], None] | None = None,
) -> list[str]:
    """
    When the planner/router omitted hex colors, build a palette from the user's text.
    Named colors are matched via _FLOW_NAMED_HEX (word-boundary, longest-first, left-to-right);
    each match maps to one hex, then the list is combined (deduping adjacent duplicates).
    Special presets: rainbow, sunset, ocean/sea. Single named color -> paired with a hue-shifted second stop.
    If no named colors match and log_fn is set, asks the LLM to infer colors from thematic references
    (games, movies, mood, etc.). Returns 2–6 #RRGGBB strings suitable for Nanoleaf flow effects.
    """
    return resolve_nanoleaf_flow_colors(user_text, log_fn)[0]


def try_handle_lighting_action(
    engine: Any,
    action: str,
    params: dict[str, Any],
    user_text: str,
    effective_text: str,
) -> str | None:
    """
    Handle routed lighting/nanoleaf actions.
    Returns a reply string when handled, else None.
    """
    if not _is_lighting_action(action):
        return None

    if _should_cancel_auto_sync_for_action(action, params):
        stop_auto_lighting_sync(log_fn=engine.log)

    if action == "lights.set_state":
        state = str(params.get("state", "")).lower()
        if state in ("on", "off", "auto"):
            engine.log(f"Turning lights {state}.")
            if state == "auto":
                start_auto_lighting_sync(log_fn=engine.log)
            else:
                toggle_all_lights("on" if state == "on" else "off")
                try:
                    if state == "on":
                        nanoleaf.turn_on()
                    else:
                        nanoleaf.turn_off()
                except Exception as e:
                    engine.log(f"Nanoleaf set_state {state} failed: {e}")
            return engine._call_model(effective_text, state)
        return engine._call_model(effective_text, None)

    if action == "lights.get_state":
        try:
            result = get_lights_state()
            light_state = result.get("state", "unknown")
            engine.log(f"The lights are {light_state}.")
            return engine._call_model(
                effective_text,
                None,
                extra_note=f"System note: The app just checked the lights; they are {light_state}.\n\n",
            )
        except LightsClientError as e:
            engine.log(f"Lights state check failed: {e}")
            return engine._call_model(
                effective_text,
                None,
                extra_note="System note: The app tried to check the lights but the request failed. Do NOT guess; tell the user the check failed and they can try again.\n\n",
            )

    if action == "lights.set_scene":
        description = str(params.get("description") or effective_text).strip()
        govee_ok = False
        govee_note = ""
        style = pick_govee_style(description, engine.log)
        if style:
            try:
                set_lights_style(state="on", **style)
                govee_ok = True
                govee_note = "Govee lights set to match the mood."
            except LightsClientError as e:
                engine.log(f"Govee set_style failed: {e}; falling back to on only.")
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
                engine.log("Govee lights turned on.")
            except LightsClientError as e:
                engine.log(f"Govee on failed: {e}")
        scenes = nanoleaf.get_scene_list()
        nanoleaf_note = ""
        chosen = pick_nanoleaf_scene(description, scenes, engine.log) if scenes else None
        if chosen:
            try:
                nanoleaf.turn_on()
                nanoleaf.set_effect(chosen)
                nanoleaf_note = f"Nanoleaf panels set to scene \"{chosen}\"."
                # If user asked for dimmer/brighter, apply brightness on top of the scene.
                desc_lower = description.lower()
                if any(w in desc_lower for w in ("dim", "bright", "brightness", "darker", "lighter", "low", "high")):
                    nl_style = pick_nanoleaf_style(description, engine.log)
                    if nl_style and nl_style.get("brightness") is not None:
                        try:
                            nanoleaf.set_brightness(nl_style["brightness"])
                            nanoleaf_note += f" Brightness set to {nl_style['brightness']}%."
                        except Exception:
                            pass
            except Exception as e:
                engine.log(f"Nanoleaf set_effect failed: {e}")
                nanoleaf_note = f"Failed to set Nanoleaf to \"{chosen}\" (panels may be unreachable)."
        else:
            # No matching scene: pick a color/brightness mood.
            style = pick_nanoleaf_style(description, engine.log)
            if style:
                try:
                    nanoleaf.turn_on()
                    if style.get("color_hex"):
                        h = style["color_hex"]
                        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
                        nanoleaf.set_color_rgb(r, g, b)
                    if style.get("brightness") is not None:
                        nanoleaf.set_brightness(style["brightness"])
                    nanoleaf_note = "Nanoleaf panels set to a custom color and brightness to match the mood."
                except Exception as e:
                    engine.log(f"Nanoleaf style failed: {e}")
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
        return engine._call_model(user_text, None, extra_note=extra_note)

    if action == "nanoleaf.set_scene":
        stop_auto_lighting_sync(log_fn=engine.log)
        description = str(params.get("description") or user_text).strip()
        scenes = nanoleaf.get_scene_list()
        if not scenes:
            return engine._call_model(
                user_text,
                None,
                extra_note="System note: The app could not read the Nanoleaf scenes list (scenes.txt). Tell the user to add scene names to lighting/nanoleaf/scenes.txt.\n\n",
            )
        chosen = pick_nanoleaf_scene(description, scenes, engine.log)
        if not chosen:
            return engine._call_model(
                user_text,
                None,
                extra_note="System note: The app could not pick a matching Nanoleaf scene from the list. Suggest the user try a different description or use 'custom' for a pulse/custom look.\n\n",
            )

        # Keep Govee in sync with the requested mood.
        govee_note = ""
        govee_style = pick_govee_style(description, engine.log)
        if not govee_style:
            nl_style = pick_nanoleaf_style(description, engine.log)
            if nl_style:
                govee_style = {
                    "color_hex": nl_style.get("color_hex"),
                    "brightness": nl_style.get("brightness"),
                }
        if govee_style:
            try:
                set_lights_style(state="on", **govee_style)
                govee_note = " Govee lights were also set to a matching color and brightness."
            except LightsClientError as e:
                engine.log(f"Govee style sync failed during nanoleaf.set_scene: {e}")
                govee_note = " Govee could not be updated."
        try:
            nanoleaf.turn_on()
            nanoleaf.set_effect(chosen)
            engine.log(f"Nanoleaf scene set to {chosen!r}.")
            return engine._call_model(
                user_text,
                None,
                extra_note=f"System note: The app has set the Nanoleaf panels to the scene \"{chosen}\".{govee_note}\n\n",
            )
        except Exception as e:
            engine.log(f"Nanoleaf set_effect failed: {e}")
            return engine._call_model(
                user_text,
                None,
                extra_note=f"System note: The app tried to set the Nanoleaf scene to \"{chosen}\" but the request failed. Tell the user to check the panels are on and reachable.{govee_note}\n\n",
            )

    if action == "nanoleaf.custom":
        description = str(params.get("description") or user_text).strip()
        dl = description.lower()
        # Router often mis-labels "create a custom animation" as custom; that needs a flow effect.
        wants_flow = (
            ("create" in dl or "new" in dl or "make" in dl or "build" in dl)
            and ("animation" in dl or "animate" in dl or "flow" in dl or "cycle" in dl)
        ) or ("custom" in dl and "animation" in dl) or (
            "rainbow" in dl and ("animation" in dl or "flow" in dl or "cycle" in dl)
        )
        if wants_flow:
            hex_list, flow_speed = resolve_nanoleaf_flow_colors(description, engine.log)
            colors_rgb: list[tuple[int, int, int]] = []
            for s in hex_list[:6]:
                try:
                    colors_rgb.append((int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)))
                except ValueError:
                    pass
            if len(colors_rgb) >= 2:
                try:
                    ok = nanoleaf.create_flow_effect(colors_rgb, flow_speed)
                    if ok:
                        engine.log("Nanoleaf flow (from custom mis-route) succeeded.")
                        persist_last_nanoleaf_flow(
                            getattr(engine, "_active_user_name", None),
                            hex_list,
                            flow_speed,
                        )
                        try:
                            set_lights_style(state="on", color_hex=hex_list[0], brightness=75)
                        except LightsClientError:
                            pass
                        return engine._call_model(
                            user_text,
                            None,
                            extra_note="System note: A flowing color animation was applied on the Nanoleaf panels.\n\n",
                        )
                    engine.log("create_flow_effect returned False (custom mis-route path).")
                except Exception as e:
                    engine.log(f"Flow from custom branch failed: {e}")

        govee_style = pick_govee_style(description, engine.log)
        scenes = nanoleaf.get_scene_list()
        nanoleaf_scene = None
        if scenes and any(w in dl for w in ("pulse", "animation", "animate", "rhythm", "beat", "moving", "dynamic")):
            nanoleaf_scene = pick_nanoleaf_scene(description, scenes, engine.log)
        nanoleaf_style = pick_nanoleaf_style(description, engine.log) if not nanoleaf_scene else None
        if not govee_style and not nanoleaf_scene and not nanoleaf_style:
            return engine._call_model(
                user_text,
                None,
                extra_note="System note: The app could not pick a custom look for the lights. Ask the user to try again or describe the look they want.\n\n",
            )
        parts = []
        if govee_style:
            try:
                set_lights_style(state="on", **govee_style)
                engine.log("Govee set to custom color/brightness.")
                parts.append("Govee lights set to a static color and brightness (Govee cannot do pulse or animation).")
            except LightsClientError as e:
                engine.log(f"Govee custom failed: {e}")
                parts.append("Govee lights could not be updated.")
        if nanoleaf_scene:
            try:
                nanoleaf.turn_on()
                nanoleaf.set_effect(nanoleaf_scene)
                engine.log(f"Nanoleaf set to scene {nanoleaf_scene!r} (animated).")
                parts.append(f"Nanoleaf panels set to the \"{nanoleaf_scene}\" scene (animated/pulse).")
            except Exception as e:
                engine.log(f"Nanoleaf set_effect failed: {e}")
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
                engine.log("Nanoleaf set to custom color/brightness.")
                parts.append("Nanoleaf panels set to a custom color and brightness (static).")
            except Exception as e:
                engine.log(f"Nanoleaf custom failed: {e}")
                parts.append("Nanoleaf panels could not be updated.")
        note = " ".join(parts)
        return engine._call_model(
            user_text,
            None,
            extra_note="System note: " + note + "\n\n",
        )

    if action == "nanoleaf.set_brightness":
        description = str(params.get("description") or user_text).strip()
        level = None
        m = re.search(r"(\d{1,3})\s*%", description)
        if m:
            level = max(0, min(100, int(m.group(1))))
        if level is None:
            style = pick_nanoleaf_style(description, engine.log)
            if style and style.get("brightness") is not None:
                level = style["brightness"]
        if level is not None:
            try:
                nanoleaf.set_brightness(level)
                engine.log(f"Nanoleaf brightness set to {level}% (no scene change).")
                return engine._call_model(
                    user_text,
                    None,
                    extra_note=f"System note: Nanoleaf panels brightness was set to {level}%. The current scene or animation was not changed.\n\n",
                )
            except Exception as e:
                engine.log(f"Nanoleaf set_brightness failed: {e}")
        return engine._call_model(
            effective_text,
            None,
            extra_note="System note: Could not set Nanoleaf brightness (panels may be unreachable).\n\n",
        )

    if action == "nanoleaf.set_state":
        state = str(params.get("state", "")).strip().lower()
        if state not in ("on", "off"):
            state = "off" if "off" in (user_text or "").lower() else "on"
        try:
            if state == "on":
                nanoleaf.turn_on()
            else:
                nanoleaf.turn_off()
            engine.log(f"Nanoleaf turned {state} (Govee unchanged).")
            return engine._call_model(
                user_text,
                None,
                extra_note=f"System note: Nanoleaf panels have been turned {state}. Govee lights were not changed.\n\n",
            )
        except Exception as e:
            engine.log(f"Nanoleaf set_state failed: {e}")
            return engine._call_model(
                user_text,
                None,
                extra_note="System note: Could not turn Nanoleaf panels on/off (panels may be unreachable).\n\n",
            )

    if action == "nanoleaf.create_animation":
        raw_colors = params.get("colors")
        desc = str(params.get("description") or user_text or "").strip()
        lower_intent = f"{desc} {user_text}".lower()
        # Speed-only follow-ups should not call Govee (avoid unnecessary API usage/call limits).
        is_speed_only_adjust = any(
            p in lower_intent
            for p in (
                "faster",
                "slower",
                "speed up",
                "slow down",
                "as fast as possible",
                "as slow as possible",
                "max speed",
                "min speed",
                "maximum speed",
                "minimum speed",
            )
        )
        hex_list: list[str] = []
        if isinstance(raw_colors, list):
            for c in raw_colors:
                s = str(c).strip()
                if s.startswith("#") and len(s) == 7:
                    hex_list.append(s.upper())
                elif len(s) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in s):
                    hex_list.append("#" + s.upper())
        inferred_colors_speed: tuple[list[str], float] | None = None
        if len(hex_list) < 2:
            inferred_colors_speed = resolve_nanoleaf_flow_colors(desc or user_text or "", engine.log)
            hex_list = inferred_colors_speed[0]
        colors_rgb: list[tuple[int, int, int]] = []
        for s in hex_list[:6]:
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
                speed = float(speed) if speed is not None else None
            except (TypeError, ValueError):
                speed = None
            if speed is None:
                if inferred_colors_speed is not None:
                    speed = inferred_colors_speed[1]
                else:
                    speed = infer_flow_speed(f"{desc} {user_text}")
            speed = max(0.1, min(5.0, round(speed, 2)))
            try:
                ok = nanoleaf.create_flow_effect(colors_rgb, speed)
                if ok:
                    engine.log("Nanoleaf create_flow_effect succeeded.")
                    persist_last_nanoleaf_flow(
                        getattr(engine, "_active_user_name", None),
                        hex_list,
                        speed,
                    )
                    first_hex = hex_list[0] if hex_list else None
                    if first_hex and not is_speed_only_adjust:
                        try:
                            set_lights_style(state="on", color_hex=first_hex, brightness=75)
                            engine.log("Govee set to match animation color.")
                        except LightsClientError:
                            pass
                    return engine._call_model(
                        user_text,
                        None,
                        extra_note="System note: A flowing animation was applied on the Nanoleaf panels with the chosen colors and speed."
                        + (" Govee lights were also set to match." if (first_hex and not is_speed_only_adjust) else "") + "\n\n",
                    )
                engine.log("Nanoleaf create_flow_effect returned False (API may have rejected the effect).")
            except Exception as e:
                engine.log(f"nanoleaf.create_animation failed: {e}")
        return engine._call_model(
            user_text,
            None,
            extra_note="System note: Could not create the Nanoleaf flow animation (device unreachable or API error). Do NOT claim success.\n\n",
        )

    return None
