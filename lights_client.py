import json
import os
from typing import Literal, Any

import requests


State = Literal["on", "off"]


BASE_URL = os.getenv("GOVEE_LIGHTS_BASE_URL", "").rstrip("/")
AUTH_TOKEN = os.getenv("GOVEE_LIGHTS_TOKEN", "")


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

