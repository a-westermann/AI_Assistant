import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

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

    pieces: List[str] = []
    if isinstance(temp_c, (int, float)):
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        pieces.append(f"{temp_f:.0f} degrees")
    if description:
        pieces.append(description)
    if isinstance(wind, (int, float)):
        # Open-Meteo returns windspeed in km/h by default.
        pieces.append(f"wind {wind:.0f} km/h")

    if not pieces:
        return "Weather is unavailable right now."
    if len(pieces) == 1:
        return pieces[0]
    if len(pieces) == 2:
        return f"{pieces[0]} and {pieces[1]}"
    # 3+ pieces: join the tail more naturally
    return f"{pieces[0]}, {pieces[1]}, and {pieces[2]}"


def _parse_time_iso(s: str) -> datetime:
    # Open-Meteo usually returns ISO strings with timezone offset when timezone=auto.
    # datetime.fromisoformat handles both "+HH:MM" and "Z" (after replacement).
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _format_time_local(dt: datetime) -> str:
    # Format like "3:00 PM" (ASCII only).
    hour12 = dt.strftime("%I").lstrip("0") or "0"
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p")
    if minute == "00":
        return f"{hour12} {ampm}"
    return f"{hour12}:{minute} {ampm}"


def get_day_weather_forecast_summary(rain_probability_threshold: float = 50.0) -> str:
    """
    Fetch a concise day forecast using Open-Meteo (no API key).

    Returns a short, user-facing summary including:
    - whether rain is likely
    - today's temperature high and when it peaks
    """
    lat, lon = _get_coords()
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation_probability"
        "&forecast_days=1"
        "&timezone=auto"
    )

    try:
        resp = requests.get(url, timeout=8)
    except Exception as e:
        raise WeatherClientError(f"Weather forecast request failed: {e}") from e

    if not resp.ok:
        raise WeatherClientError(f"Weather forecast HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data: Dict[str, Any] = resp.json()
    except Exception as e:
        raise WeatherClientError(f"Bad weather forecast JSON: {e}") from e

    hourly = data.get("hourly") or {}
    times: List[str] = hourly.get("time") or []
    temps_c: List[Any] = hourly.get("temperature_2m") or []
    pp: List[Any] = hourly.get("precipitation_probability") or []

    if not (times and temps_c and pp) or not (len(times) == len(temps_c) == len(pp)):
        raise WeatherClientError("Weather forecast missing hourly fields.")

    d0 = _parse_time_iso(times[0]).date()

    today_indices = []
    for i, t in enumerate(times):
        try:
            if _parse_time_iso(t).date() == d0:
                today_indices.append(i)
        except Exception:
            continue

    if not today_indices:
        raise WeatherClientError("Weather forecast had no hourly points for today.")

    # Temperature high and time of peak (max temp).
    best_i = max(today_indices, key=lambda i: temps_c[i] if isinstance(temps_c[i], (int, float)) else -1e9)
    best_temp_c = temps_c[best_i]
    best_temp_f = best_temp_c * 9.0 / 5.0 + 32.0
    best_time = _format_time_local(_parse_time_iso(times[best_i]))

    # Rain probability window and peak time.
    rain_indices = [
        i
        for i in today_indices
        if isinstance(pp[i], (int, float)) and pp[i] >= rain_probability_threshold
    ]

    if not rain_indices:
        return f"Today should reach a high of {best_temp_f:.0f} degrees around {best_time}. Rain doesn't look likely."

    rain_start_i = min(rain_indices)
    rain_end_i = max(rain_indices)
    rain_start_time = _format_time_local(_parse_time_iso(times[rain_start_i]))
    rain_end_time = _format_time_local(_parse_time_iso(times[rain_end_i]))

    peak_rain_i = max(rain_indices, key=lambda i: pp[i] if isinstance(pp[i], (int, float)) else -1e9)
    peak_rain_time = _format_time_local(_parse_time_iso(times[peak_rain_i]))
    peak_rain_pp = pp[peak_rain_i]

    if rain_start_time == rain_end_time:
        rain_window = f"around {rain_start_time}"
    else:
        rain_window = f"from {rain_start_time} to {rain_end_time}"

    return (
        f"Today should reach a high of {best_temp_f:.0f} degrees around {best_time}. "
        f"Rain is likely {rain_window}. "
        f"The wettest part should peak around {peak_rain_time} (about {peak_rain_pp:.0f}%)."
    )


def _weather_code_description(code: int | None) -> str:
    return {
        0: "Clear",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Fog",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Snow",
        75: "Heavy snow",
    }.get(code, "Mixed")


def _weather_code_icon_key(code: int | None) -> str:
    if code in (0, 1):
        return "clear"
    if code in (2, 3, 45, 48):
        return "cloud"
    if code in (51, 53, 55, 61, 63, 65):
        return "rain"
    if code in (71, 73, 75):
        return "snow"
    return "mixed"


def get_weather_ui_payload(hours: int = 12) -> Dict[str, Any]:
    """
    Structured weather payload for UI rendering.
    Includes current conditions + next hourly points.
    """
    lat, lon = _get_coords()
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current_weather=true"
        "&hourly=temperature_2m,precipitation_probability,weathercode"
        "&forecast_days=1"
        "&timezone=auto"
    )
    try:
        resp = requests.get(url, timeout=8)
    except Exception as e:
        raise WeatherClientError(f"Weather UI request failed: {e}") from e
    if not resp.ok:
        raise WeatherClientError(f"Weather UI HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data: Dict[str, Any] = resp.json()
    except Exception as e:
        raise WeatherClientError(f"Bad weather UI JSON: {e}") from e

    cw = data.get("current_weather") or {}
    current_temp_c = cw.get("temperature")
    current_code = cw.get("weathercode")
    current_time = cw.get("time")
    current_temp_f = (
        int(round(current_temp_c * 9.0 / 5.0 + 32.0))
        if isinstance(current_temp_c, (int, float))
        else None
    )

    hourly = data.get("hourly") or {}
    times: List[str] = hourly.get("time") or []
    temps_c: List[Any] = hourly.get("temperature_2m") or []
    precip: List[Any] = hourly.get("precipitation_probability") or []
    codes: List[Any] = hourly.get("weathercode") or []

    if not (times and temps_c and precip and codes):
        raise WeatherClientError("Weather UI payload missing hourly fields.")

    # Start from "now" (or nearest future hourly point), not midnight.
    start_idx = 0
    now_dt = None
    if isinstance(current_time, str):
        try:
            now_dt = _parse_time_iso(current_time)
        except Exception:
            now_dt = None
    if now_dt is not None:
        for i, t_iso in enumerate(times):
            try:
                if _parse_time_iso(t_iso) >= now_dt:
                    start_idx = i
                    break
            except Exception:
                continue
    elif isinstance(current_time, str) and current_time in times:
        # Exact fallback for any payloads that already align exactly.
        start_idx = times.index(current_time)

    points: List[Dict[str, Any]] = []
    end_idx = min(len(times), start_idx + max(1, int(hours)))
    for i in range(start_idx, end_idx):
        t_iso = times[i]
        try:
            dt = _parse_time_iso(t_iso)
            time_label = _format_time_local(dt)
        except Exception:
            time_label = t_iso[-5:]

        t_c = temps_c[i]
        t_f = int(round(t_c * 9.0 / 5.0 + 32.0)) if isinstance(t_c, (int, float)) else None
        p = precip[i]
        p_pct = int(round(float(p))) if isinstance(p, (int, float)) else 0
        code_i = codes[i] if isinstance(codes[i], int) else None

        points.append(
            {
                "time": time_label,
                "temp_f": t_f,
                "precip_pct": p_pct,
                "icon": _weather_code_icon_key(code_i),
            }
        )

    return {
        "location": "St. Charles",
        "current_temp_f": current_temp_f,
        "current_desc": _weather_code_description(current_code if isinstance(current_code, int) else None),
        "current_icon": _weather_code_icon_key(current_code if isinstance(current_code, int) else None),
        "hourly": points,
    }

