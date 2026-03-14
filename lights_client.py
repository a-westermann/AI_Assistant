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
    if "state" in data and data["state"] in ("on", "off"):
        return {"state": data["state"], "success": True}
    if "lights_on" in data:
        return {"state": "on" if data["lights_on"] else "off", "success": True}
    raise LightsClientError(f"Lights API did not return state: {data}")


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
