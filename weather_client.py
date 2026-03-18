import os
from typing import Any, Dict

import requests


class WeatherClientError(Exception):
    pass


def _get_coords() -> tuple[float, float]:
    """
    Return (lat, lon) for St. Charles, MO (63303).
    Hard-coded to the user's location so we don't depend on env vars.
    """
    # Approximate coordinates for St. Charles, MO 63303
    return 38.7839, -90.5395


def get_current_weather_summary() -> str:
    """
    Fetch a short, user-facing weather summary using the free Open-Meteo API (no API key).
    """
    lat, lon = _get_coords()
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )
    try:
        resp = requests.get(url, timeout=5)
    except Exception as e:
        raise WeatherClientError(f"Weather request failed: {e}") from e

    if not resp.ok:
        raise WeatherClientError(f"Weather HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data: Dict[str, Any] = resp.json()
    except Exception as e:
        raise WeatherClientError(f"Bad weather JSON: {e}") from e

    cw = data.get("current_weather") or {}
    temp_c = cw.get("temperature")
    wind = cw.get("windspeed")
    code = cw.get("weathercode")

    # Simple mapping for the most common codes
    description = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "foggy",
        51: "light drizzle",
        53: "drizzle",
        55: "heavy drizzle",
        61: "light rain",
        63: "rain",
        65: "heavy rain",
        71: "light snow",
        73: "snow",
        75: "heavy snow",
    }.get(code, "mixed conditions")

    pieces = []
    if isinstance(temp_c, (int, float)):
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        # ASCII only: keep temperatures as plain numbers followed by F.
        pieces.append(f"{temp_f:.0f}F")
    pieces.append(description)
    if isinstance(wind, (int, float)):
        # Open-Meteo returns windspeed in km/h by default.
        pieces.append(f"wind {wind:.0f} km/h")

    return ", ".join(pieces) if pieces else "Weather data is currently unavailable."

