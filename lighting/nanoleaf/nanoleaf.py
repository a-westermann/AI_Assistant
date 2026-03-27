import os
import time
import colorsys

import requests

# Last create_flow_effect attempt (read by API /chat and /lights/diagnostic).
_last_flow_attempt: dict | None = None

NANOLEAF_IP = "192.168.0.188"
TOKEN_FILE = "token.txt"
SCENES_FILE = "scenes.txt"


def _token_path() -> str:
    """Return absolute path to the auth token file (token.txt next to this script)."""
    here = os.path.dirname(__file__)
    return os.path.join(here, TOKEN_FILE)


def get_token() -> str:
    with open(_token_path(), "r", encoding="utf-8") as f:
        return f.read().strip()


def get_last_flow_attempt() -> dict | None:
    """Snapshot from the most recent create_flow_effect call (ok, detail, mode, color_count, time_unix)."""
    return _last_flow_attempt


def _set_flow_attempt(
    ok: bool,
    *,
    detail: str = "",
    color_count: int = 0,
    mode: str = "",
) -> None:
    global _last_flow_attempt
    _last_flow_attempt = {
        "ok": ok,
        "detail": (detail or "")[:500],
        "color_count": color_count,
        "mode": mode,
        "time_unix": time.time(),
    }


def _put_state(payload: dict) -> requests.Response:
    """Low-level helper to PUT /state with a partial payload."""
    token = get_token()
    url = f"http://{NANOLEAF_IP}:16021/api/v1/{token}/state"
    response = requests.put(url, json=payload, timeout=5)
    if response.status_code != 204:
        print("Nanoleaf state update failed:", response.status_code, response.text)
    return response


def send_power_state(state: bool) -> None:
    resp = _put_state({"on": {"value": state}})
    if resp.status_code == 204:
        print("Lights ON" if state else "Lights OFF")


def set_brightness(level: int) -> None:
    """
    Set global brightness (0–100).
    Values outside the range are clamped.
    """
    level = max(0, min(100, int(level)))
    _put_state({"brightness": {"value": level}})
    print(f"Brightness set to {level}")


def set_color_hs(hue: int, sat: int) -> None:
    """
    Set color using Nanoleaf's hue/sat model.
    - hue: 0–360
    - sat: 0–100
    """
    hue = max(0, min(360, int(hue)))
    sat = max(0, min(100, int(sat)))
    _put_state({"hue": {"value": hue}, "sat": {"value": sat}})
    print(f"Color set to hue={hue}, sat={sat}")


def set_color_rgb(r: int, g: int, b: int) -> None:
    """
    Convenience helper: set color from RGB (0–255 each).
    Converts to HSV and sends hue/sat to Nanoleaf.
    """
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    # colorsys uses 0–1 floats; returns h,s,v in 0–1
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue = int(h * 360)
    sat = int(s * 100)
    set_color_hs(hue, sat)


def _scenes_path() -> str:
    """Return absolute path to scenes.txt (next to this script)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, SCENES_FILE)


def get_scene_list() -> list[str]:
    """
    Read scene names from scenes.txt (one per line).
    Returns list of non-empty stripped names; empty list if file missing or unreadable.
    """
    path = _scenes_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def set_effect(name: str) -> None:
    """
    Select a built-in or user-created effect/scene by name.
    The effect must already exist on the controller.
    """
    token = get_token()
    url = f"http://{NANOLEAF_IP}:16021/api/v1/{token}/effects/select"
    payload = {"select": str(name)}
    response = requests.put(url, json=payload, timeout=5)
    if response.status_code == 200:
        print(f"Effect '{name}' selected")
    else:
        print("Failed to select effect:", response.status_code, response.text)


def get_selected_effect() -> str | None:
    """
    Return currently selected effect/scene name from Nanoleaf, or None on failure.
    """
    token = get_token()
    url = f"http://{NANOLEAF_IP}:16021/api/v1/{token}/effects/select"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, dict):
            sel = data.get("select")
            if isinstance(sel, str) and sel.strip():
                return sel.strip()
        return None
    except Exception:
        return None


def _get_layout() -> dict | None:
    """GET panel layout (positions and panel IDs). Returns None on failure."""
    token = get_token()
    url = f"http://{NANOLEAF_IP}:16021/api/v1/{token}/panelLayout/layout"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def _write_effect(write_payload: dict) -> tuple[bool, str]:
    """Send a custom effect to the device. Returns (success, error_detail)."""
    token = get_token()
    url = f"http://{NANOLEAF_IP}:16021/api/v1/{token}/effects"
    try:
        resp = requests.put(url, json={"write": write_payload}, timeout=10)
        if resp.status_code in (200, 204):
            return True, ""
        err = f"HTTP {resp.status_code}: {(resp.text or '')[:400]}"
        print("Nanoleaf write_effect failed:", err)
        return False, err
    except Exception as e:
        print("Nanoleaf write_effect error:", e)
        return False, str(e)


def _rgb_to_hsb_palette_entry(r: int, g: int, b: int) -> dict:
    """Convert RGB 0-255 to Nanoleaf palette entry: hue 0-360, saturation 0-100, brightness 0-100."""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return {
        "hue": int(round(h * 360)) % 360,
        "saturation": int(round(s * 100)),
        "brightness": max(1, int(round(v * 100))),
    }


def create_flow_effect(colors_rgb: list[tuple[int, int, int]], speed: float = 1.0) -> bool:
    """
    Create and display a flowing animation that cycles through the given RGB colors on ALL panels, with looping.
    Uses the Nanoleaf REST API write effect (palette-based, loop enabled). Does not use nanoleafapi.
    - colors_rgb: list of (r, g, b) tuples, each 0-255. At least 2 colors.
    - speed: transition-like value; lower is faster, higher is slower.
      Example: 0.1 = very fast, 1.0 = medium, 5.0 = very slow.
    Returns True if the effect was created and displayed, False otherwise.
    """
    if not colors_rgb or len(colors_rgb) < 2:
        print("create_flow_effect needs at least 2 colors.")
        _set_flow_attempt(False, detail="Need at least 2 colors.", color_count=0, mode="skipped")
        return False
    rgb_list = []
    for c in colors_rgb:
        r, g, b = int(c[0]), int(c[1]), int(c[2])
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        rgb_list.append((r, g, b))
    speed = round(float(speed), 2)
    speed = max(0.1, min(5.0, speed))
    # Normalize with inverse semantics: 1.0 => fastest end, 0.0 => slowest end.
    speed_norm = (5.0 - speed) / 4.9
    # Convert to HSB palette (Nanoleaf format)
    palette = [_rgb_to_hsb_palette_entry(r, g, b) for r, g, b in rgb_list]
    # Piecewise timing map for HSB random fallback.
    # This creates intentionally large visual differences between speed bands.
    if speed <= 0.2:
        trans_min, delay_min = 10, 8
    elif speed <= 0.5:
        trans_min, delay_min = 20, 12
    elif speed <= 1.0:
        trans_min, delay_min = 36, 24
    elif speed <= 2.0:
        trans_min, delay_min = 62, 45
    elif speed <= 3.5:
        trans_min, delay_min = 95, 72
    else:
        trans_min, delay_min = 120, 95
    trans_max = min(140, trans_min + 30)
    delay_max = min(120, delay_min + 26)
    turn_on()
    # Use a unique name per speed update so controllers don't reuse stale timing.
    effect_name = f"Custom flow {int(time.time() * 1000)}"
    # REST write payload path (primary): keep speed semantics explicit in payload detail.
    write_payload = {
        "command": "display",
        "animName": effect_name,
        "animType": "random",
        "colorType": "HSB",
        "animData": None,
        "palette": palette,
        # Keep the raw transition-like value visible and pass it through;
        # some controllers/firmware honor this directly.
        "speed": speed,
        "brightnessRange": {"minValue": 50, "maxValue": 100},
        "transTime": {"minValue": trans_min, "maxValue": trans_max},
        "delayTime": {"minValue": delay_min, "maxValue": delay_max},
        "loop": True,
    }
    ok_hsb, err_hsb = _write_effect(write_payload)
    if ok_hsb:
        print("Flow effect created and displayed (looping, REST write).")
        _set_flow_attempt(
            True,
            detail=(
                f"speed={speed}, trans={trans_min}-{trans_max}, delay={delay_min}-{delay_max}"
            ),
            color_count=len(rgb_list),
            mode="rest_hsb_random",
        )
        return True
    _set_flow_attempt(False, detail=f"rest_hsb: {err_hsb}", color_count=len(rgb_list), mode="failed")
    return False


def turn_on() -> None:
    send_power_state(True)


def turn_off() -> None:
    send_power_state(False)


if __name__ == "__main__":
    # simple manual test when run directly
    print("Turning lights ON...")
    turn_on()

    print("Setting brightness to 60...")
    set_brightness(60)

    print("Setting color to blue via RGB...")
    set_color_rgb(0, 128, 255)

    input("Press Enter to turn lights OFF...")

    turn_off()