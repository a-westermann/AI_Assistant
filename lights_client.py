import json
import os
from typing import Literal, Any

import requests


State = Literal["on", "off"]


BASE_URL = os.getenv("GOVEE_LIGHTS_BASE_URL", "").rstrip("/")
AUTH_TOKEN = os.getenv("GOVEE_LIGHTS_TOKEN", "")
# Optional: override status path if your server uses something other than "status" (e.g. "state")
STATUS_PATH = (os.getenv("GOVEE_LIGHTS_STATUS_PATH") or "status").strip("/") or "status"


class LightsClientError(Exception):
    """Raised when the lights API returns an error or is misconfigured."""


def _require_config() -> None:
    if not BASE_URL or not AUTH_TOKEN:
        raise LightsClientError(
            "Lights client not configured. "
            "Set GOVEE_LIGHTS_BASE_URL and GOVEE_LIGHTS_TOKEN environment variables."
        )


def _auth_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Auth-Token": AUTH_TOKEN,
    }


def get_lights_state() -> dict[str, Any]:
    """
    Get the current lights state from your Govee automation service.

    Expects an endpoint that returns the current state, e.g.:

        GET /status/  (or set GOVEE_LIGHTS_STATUS_PATH to your path, e.g. "state")
        Headers: X-Auth-Token: <token>
        Response: {"state": "on" | "off"}  OR  {"lights_on": true | false}

    Returns dict with key "state" ("on" or "off") and optionally "success".
    """
    _require_config()

    url = f"{BASE_URL}/{STATUS_PATH}/"
    headers = _auth_headers()

    response = requests.get(url, headers=headers, timeout=15)
    try:
        data = response.json()
    except Exception:
        if response.status_code == 404:
            raise LightsClientError(
                f"Status endpoint returned 404. We're calling: GET {url} "
                f"Make sure that URL exists on your lights server (same app as toggle_light). "
                f"If your path is different, set GOVEE_LIGHTS_STATUS_PATH (e.g. to 'state')."
            )
        raise LightsClientError(f"Unexpected response from lights API: {response.text[:200]}")

    if not response.ok:
        raise LightsClientError(f"Lights API error: {data}")

    # Support both {"state": "on"|"off"} and {"lights_on": true|false}
    state: str | None = None
    if "state" in data and data["state"] in ("on", "off"):
        state = str(data["state"])
    elif "lights_on" in data:
        state = "on" if data["lights_on"] else "off"

    if state is None:
        raise LightsClientError(f"Lights API did not return state: {data}")

    def _coerce_int(v: Any) -> int | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except Exception:
                return None
        if isinstance(v, str):
            vv = v.strip()
            if vv.isdigit():
                try:
                    return int(vv)
                except Exception:
                    return None
        if isinstance(v, dict):
            for kk in ("value", "current", "brightness", "percent", "brightnessPercent"):
                if kk in v:
                    return _coerce_int(v.get(kk))
        return None

    # Optional: brightness (many status payloads include it, sometimes nested).
    brightness: int | None = None
    for k in (
        "brightness",
        "brightness_level",
        "current_brightness",
        "currentBrightness",
        "brightnessPercent",
        "brightness_percent",
        "dimming",
        "dimmer",
    ):
        if k in data:
            b = _coerce_int(data.get(k))
            if b is not None:
                brightness = max(0, min(100, b))
                break

    # Optional: mode (auto vs manual).
    mode: str | None = None
    for k in (
        "mode",
        "current_mode",
        "controlMode",
        "lightingMode",
        "workMode",
        "lighting_mode",
        "effect_mode",
    ):
        if k in data and isinstance(data.get(k), str):
            mode = data.get(k)
            break
    if mode is None:
        for k in ("auto", "automatic", "is_auto", "isAuto"):
            if k in data:
                v = data.get(k)
                if isinstance(v, bool) and v:
                    mode = "auto"
                    break

    # Optional: color information (depends on your status endpoint schema).
    color_hex: str | None = None
    for k in ("color", "color_hex", "colorHex", "colorCode", "colorCodeHex"):
        v = data.get(k)
        if isinstance(v, str) and v.strip().startswith("#"):
            color_hex = v.strip()
            break
        if isinstance(v, dict):
            # e.g. {"hex":"#FFFFFF"} or {"value":"#FFFFFF"}
            for kk in ("hex", "value", "color"):
                vv = v.get(kk) if isinstance(v, dict) else None
                if isinstance(vv, str) and vv.strip().startswith("#"):
                    color_hex = vv.strip()
                    break
        if color_hex:
            break

    # Optional: color temperature (Kelvin) if present.
    color_temp_k: int | None = None
    for k in (
        "color_temp_k",
        "colorTempK",
        "colorTemp",
        "temperature",
        "color_temperature",
        "colorTemperature",
    ):
        if k in data:
            ct = _coerce_int(data.get(k))
            if ct is not None:
                # Clamp to plausible warm/cool LED range.
                color_temp_k = max(0, min(20000, ct))
                break

    out: dict[str, Any] = {"state": state, "success": True}
    if brightness is not None:
        out["brightness"] = brightness
    if mode is not None:
        out["mode"] = mode
    if color_hex is not None:
        out["color_hex"] = color_hex
    if color_temp_k is not None:
        out["color_temp_k"] = color_temp_k
    return out


def toggle_all_lights(state: State) -> dict[str, Any]:
    """
    Toggle all lights on or off via your existing Govee automation service.

    This assumes you have an HTTP endpoint compatible with the Django view
    `toggle_light` in your `govee_bulb_automation` project, i.e. it accepts:

        POST /toggle_light/
        Headers: X-Auth-Token: <token>
        Body:   {"state": "on" | "off"}
    """
    if state not in ("on", "off"):
        raise ValueError(f"Invalid state {state!r}; expected 'on' or 'off'.")

    _require_config()

    url = f"{BASE_URL}/toggle_light/"
    payload = {"state": state}
    headers = _auth_headers()

    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=15)
    try:
        data = response.json()
    except Exception:
        raise LightsClientError(f"Unexpected response from lights API: {response.text}")

    if not response.ok or not data.get("success", False):
        raise LightsClientError(f"Lights API error: {data}")

    return data


def set_lights_auto() -> dict[str, Any]:
    """
    Put the lights into automatic mode via your existing web app.

    This assumes an authenticated endpoint at:

        POST /auto
        Headers: X-Auth-Token: <token>
    """
    _require_config()

    url = f"{BASE_URL}/auto"
    headers = _auth_headers()

    response = requests.post(url, headers=headers, timeout=15)
    try:
        data = response.json()
    except Exception:
        raise LightsClientError(f"Unexpected response from lights API: {response.text}")

    if not response.ok or not data.get("success", False):
        raise LightsClientError(f"Lights API error: {data}")

    return data


def set_lights_style(
    *,
    state: State = "on",
    color_hex: str | None = None,
    color_temp_k: int | None = None,
    brightness: int | None = None,
) -> dict[str, Any]:
    """
    Set Govee lights on/off plus optional color, color temperature, and brightness.

    Calls govee_bulb_automation endpoints in sequence:
        POST /toggle_light/  {"state": "on"|"off"}
        POST /set_color/     {"color": "#RRGGBB"}   (optional)
        POST /set_temperature/ {"temperature": <kelvin>} (optional)
        POST /set_brightness/  {"brightness": 0-100}     (optional)
    """
    _require_config()
    headers = _auth_headers()

    def _parse_json_response(response: requests.Response, label: str) -> dict:
        """Parse JSON or raise a clear error including status code when server returns HTML (e.g. 500)."""
        if response.status_code >= 400:
            try:
                data = response.json()
            except Exception:
                raise LightsClientError(
                    f"Lights server returned {response.status_code} ({label}); body is not JSON. "
                    f"Check server logs. First 300 chars: {response.text.strip()[:300]}"
                )
            raise LightsClientError(f"Lights API error ({label}): {data}")
        try:
            return response.json()
        except Exception:
            raise LightsClientError(
                f"Lights API returned {response.status_code} but body is not JSON ({label}). "
                f"First 300 chars: {response.text.strip()[:300]}"
            )

    if state == "on":
        url = f"{BASE_URL}/toggle_light/"
        response = requests.post(
            url, data=json.dumps({"state": "on"}), headers=headers, timeout=15
        )
        data = _parse_json_response(response, "toggle")
        if not data.get("success", False):
            raise LightsClientError(f"Lights API error (toggle): {data}")

    if color_hex is not None:
        hex_val = str(color_hex).strip()
        if not hex_val.startswith("#"):
            hex_val = "#" + hex_val
        url = f"{BASE_URL}/set_color/"
        response = requests.post(
            url, data=json.dumps({"color": hex_val}), headers=headers, timeout=15
        )
        data = _parse_json_response(response, "set_color")
        if not data.get("success", False):
            raise LightsClientError(f"Lights API error (set_color): {data}")

    if color_temp_k is not None:
        temp = max(2200, min(6500, int(color_temp_k)))
        url = f"{BASE_URL}/set_temperature/"
        response = requests.post(
            url, data=json.dumps({"temperature": temp}), headers=headers, timeout=15
        )
        data = _parse_json_response(response, "set_temperature")
        if not data.get("success", False):
            raise LightsClientError(f"Lights API error (set_temperature): {data}")

    if brightness is not None:
        b = max(0, min(100, int(brightness)))
        url = f"{BASE_URL}/set_brightness/"
        response = requests.post(
            url, data=json.dumps({"brightness": b}), headers=headers, timeout=15
        )
        data = _parse_json_response(response, "set_brightness")
        if not data.get("success", False):
            raise LightsClientError(f"Lights API error (set_brightness): {data}")

    return {"success": True}


if __name__ == "__main__":
    # Run: python lights_client.py   (with env vars set) to see the status URL and test it
    print("BASE_URL:", BASE_URL or "(not set)")
    print("Status URL:", f"{BASE_URL}/{STATUS_PATH}/" if BASE_URL else "N/A")
    if not BASE_URL or not AUTH_TOKEN:
        print("Set GOVEE_LIGHTS_BASE_URL and GOVEE_LIGHTS_TOKEN, then try again.")
        raise SystemExit(1)
    try:
        out = get_lights_state()
        print("OK:", out)
    except LightsClientError as e:
        print("Error:", e)
        raise SystemExit(1)
