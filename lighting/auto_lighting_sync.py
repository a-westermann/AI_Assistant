"""
Keeps Nanoleaf in sync with Govee auto mode profile.
"""

from __future__ import annotations

import threading
import time
import math
from datetime import datetime, timedelta
from typing import Callable

import requests

from .lights_client import exit_lights_auto_mode, set_lights_auto, LightsClientError
from .nanoleaf import nanoleaf
from misc_tools.weather_client import _get_coords

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _calculate_light_temperature(
    sunrise_dt: datetime,
    sunset_dt: datetime,
    current_dt: datetime | None = None,
    min_temp: int = 2700,
    max_temp: int = 6500,
) -> int:
    now = current_dt or datetime.utcnow()
    sunrise_start = sunrise_dt - timedelta(minutes=30)
    sunrise_end = sunrise_dt + timedelta(hours=3)
    sunset_start = sunset_dt - timedelta(hours=3)
    sunset_end = sunset_dt - timedelta(minutes=30)
    if now < sunrise_start:
        return min_temp
    if sunrise_start <= now < sunrise_end:
        ratio = (now - sunrise_start).total_seconds() / (sunrise_end - sunrise_start).total_seconds()
        return int(min_temp + (max_temp - min_temp) * ratio)
    if sunrise_end <= now < sunset_start:
        return max_temp
    if sunset_start <= now < sunset_end:
        ratio = (now - sunset_start).total_seconds() / (sunset_end - sunset_start).total_seconds()
        return int(max_temp - (max_temp - min_temp) * ratio)
    return min_temp


def _calculate_brightness(
    sunrise_dt: datetime,
    sunset_dt: datetime,
    current_dt: datetime | None = None,
    min_brightness: int = 5,
    max_brightness: int = 75,
) -> int:
    now = current_dt or datetime.utcnow()
    sunrise_start = sunrise_dt - timedelta(minutes=30)
    sunrise_end = sunrise_dt + timedelta(hours=3)
    sunset_start = sunset_dt - timedelta(hours=3)
    sunset_end = sunset_dt - timedelta(minutes=30)
    if now < sunrise_start:
        return min_brightness
    if sunrise_start <= now < sunrise_end:
        ratio = (now - sunrise_start).total_seconds() / (sunrise_end - sunrise_start).total_seconds()
        return round(min_brightness + (max_brightness - min_brightness) * ratio)
    if sunrise_end <= now < sunset_start:
        return max_brightness
    if sunset_start <= now < sunset_end:
        ratio = (now - sunset_start).total_seconds() / (sunset_end - sunset_start).total_seconds()
        return round(max_brightness - (max_brightness - min_brightness) * ratio)
    return min_brightness


def _kelvin_to_rgb(temp_k: int) -> tuple[int, int, int]:
    # Approximate conversion for white point on Nanoleaf.
    t = max(1000, min(40000, int(temp_k))) / 100.0
    if t <= 66:
        red = 255
        green = 99.4708025861 * (t ** 0.0) if t <= 0 else 99.4708025861 * math.log(t) - 161.1195681661
        blue = 0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        red = 329.698727446 * ((t - 60) ** -0.1332047592)
        green = 288.1221695283 * ((t - 60) ** -0.0755148492)
        blue = 255
    r = int(max(0, min(255, red)))
    g = int(max(0, min(255, green)))
    b = int(max(0, min(255, blue)))
    return r, g, b


def _fetch_sun_times_utc() -> tuple[datetime, datetime]:
    lat, lon = _get_coords()
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=sunrise,sunset"
        "&forecast_days=1"
        "&timezone=UTC"
    )
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily") or {}
    sunrise = (daily.get("sunrise") or [None])[0]
    sunset = (daily.get("sunset") or [None])[0]
    if not sunrise or not sunset:
        raise RuntimeError("Missing sunrise/sunset data")
    if str(sunrise).endswith("Z"):
        sunrise = str(sunrise)[:-1] + "+00:00"
    if str(sunset).endswith("Z"):
        sunset = str(sunset)[:-1] + "+00:00"
    return datetime.fromisoformat(str(sunrise)), datetime.fromisoformat(str(sunset))


def _apply_nanoleaf_auto_profile(log_fn: Callable[[str], None] | None = None) -> dict:
    sunrise_dt, sunset_dt = _fetch_sun_times_utc()
    now = datetime.utcnow().replace(tzinfo=sunrise_dt.tzinfo)
    temp_k = _calculate_light_temperature(sunrise_dt, sunset_dt, now)
    brightness = _calculate_brightness(sunrise_dt, sunset_dt, now)
    r, g, b = _kelvin_to_rgb(temp_k)
    nanoleaf.turn_on()
    nanoleaf.set_color_rgb(r, g, b)
    nanoleaf.set_brightness(int(brightness))
    if log_fn:
        log_fn(f"Nanoleaf auto sync applied: {temp_k}K, {brightness}%")
    return {"temperature_k": int(temp_k), "brightness": int(brightness)}


def _nanoleaf_auto_worker(log_fn: Callable[[str], None] | None = None) -> None:
    if log_fn:
        log_fn("Nanoleaf auto sync worker started.")
    while not _stop_event.is_set():
        try:
            _apply_nanoleaf_auto_profile(log_fn=log_fn)
        except Exception as e:
            if log_fn:
                log_fn(f"Nanoleaf auto sync failed: {e}")
        _stop_event.wait(60.0)
    if log_fn:
        log_fn("Nanoleaf auto sync worker stopped.")


def start_auto_lighting_sync(log_fn: Callable[[str], None] | None = None) -> dict:
    """
    Enable Govee auto mode, apply Nanoleaf matching profile now,
    and keep Nanoleaf updated every 60s.
    """
    result = set_lights_auto()
    nl = _apply_nanoleaf_auto_profile(log_fn=log_fn)
    with _lock:
        global _thread
        if _thread is None or not _thread.is_alive():
            _stop_event.clear()
            _thread = threading.Thread(
                target=_nanoleaf_auto_worker,
                kwargs={"log_fn": log_fn},
                daemon=True,
                name="nanoleaf-auto-sync",
            )
            _thread.start()
    out = dict(result)
    out["nanoleaf"] = {"success": True, **nl}
    out["nanoleaf_auto_sync"] = True
    return out


def stop_auto_lighting_sync(log_fn: Callable[[str], None] | None = None) -> None:
    with _lock:
        _stop_event.set()
    if log_fn:
        log_fn("Requested stop for Nanoleaf auto sync.")
    # Govee may still be in /auto until we POST /manual/; Nanoleaf-only scenes never hit set_color/toggle.
    if exit_lights_auto_mode():
        if log_fn:
            log_fn("Govee left automatic mode (manual scene / override).")


def is_auto_lighting_sync_live() -> bool:
    """Return whether the Nanoleaf auto-sync worker is currently alive."""
    with _lock:
        return _thread is not None and _thread.is_alive() and not _stop_event.is_set()
