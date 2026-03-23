"""
Lighting action handlers extracted from assistant_engine.py.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm import ask_lmstudio
from lights_client import (
    get_lights_state,
    toggle_all_lights,
    LightsClientError,
    set_lights_auto,
    set_lights_style,
)
from nanoleaf import nanoleaf
from auto_lighting_sync import start_auto_lighting_sync, stop_auto_lighting_sync


def _is_lighting_action(action: str) -> bool:
    a = (action or "").strip().lower()
    return a.startswith("lights.") or a.startswith("nanoleaf.")


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

    if action == "lights.set_state":
        state = str(params.get("state", "")).lower()
        if state in ("on", "off", "auto"):
            engine.log(f"Turning lights {state}.")
            if state == "auto":
                start_auto_lighting_sync(log_fn=engine.log)
            else:
                stop_auto_lighting_sync(log_fn=engine.log)
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
                extra_note="System note: The app could not read the Nanoleaf scenes list (scenes.txt). Tell the user to add scene names to nanoleaf/scenes.txt.\n\n",
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
        govee_style = pick_govee_style(description, engine.log)
        scenes = nanoleaf.get_scene_list()
        nanoleaf_scene = None
        if scenes and any(w in description.lower() for w in ("pulse", "animation", "animate", "rhythm", "beat", "moving", "dynamic")):
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
                        engine.log("Nanoleaf create_flow_effect succeeded.")
                        first_hex = raw_colors[0] if raw_colors else None
                        if first_hex:
                            try:
                                set_lights_style(state="on", color_hex=first_hex, brightness=75)
                                engine.log("Govee set to match animation color.")
                            except LightsClientError:
                                pass
                        return engine._call_model(
                            user_text,
                            None,
                            extra_note="System note: A new flowing animation was created on the Nanoleaf panels with the chosen colors and speed."
                            + (" Govee lights were also set to match." if first_hex else "") + "\n\n",
                        )
                except Exception as e:
                    engine.log(f"nanoleaf.create_animation failed: {e}")
        return engine._call_model(
            user_text,
            None,
            extra_note="System note: Could not create the animation (need at least 2 valid hex colors in params.colors, e.g. [\"#FF0000\", \"#0000FF\"]).\n\n",
        )

    return None
