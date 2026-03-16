"""
Pomodoro focus timer with ambient light integration.

Manages focus/break cycles and shifts Govee + Nanoleaf lights to match each phase.
Thread-safe singleton so the timer persists across API requests.
"""

import datetime
import threading
from typing import Any, Callable, Optional

from lights_client import set_lights_style, LightsClientError
from nanoleaf import nanoleaf


# -- Phase durations (minutes) ------------------------------------------------

FOCUS_MINUTES = 25
SHORT_BREAK_MINUTES = 5
LONG_BREAK_MINUTES = 15
POMODOROS_BEFORE_LONG_BREAK = 4

# -- Light presets per phase ---------------------------------------------------

LIGHT_PRESETS = {
    "focus": {
        "govee": {"color_hex": "#4488FF", "color_temp_k": 5500, "brightness": 80},
        "nanoleaf_rgb": (68, 136, 255),
        "nanoleaf_brightness": 75,
        "nanoleaf_scene": None,
    },
    "short_break": {
        "govee": {"color_hex": "#FF8844", "color_temp_k": 2700, "brightness": 50},
        "nanoleaf_rgb": (255, 136, 68),
        "nanoleaf_brightness": 50,
        "nanoleaf_scene": "Inner Peace",
    },
    "long_break": {
        "govee": {"color_hex": "#44BB88", "color_temp_k": 3000, "brightness": 40},
        "nanoleaf_rgb": (68, 187, 136),
        "nanoleaf_brightness": 40,
        "nanoleaf_scene": "Northern Lights",
    },
}


class PomodoroTimer:
    """Thread-safe pomodoro timer with automatic light transitions."""

    def __init__(self, log_fn: Optional[Callable[[str], None]] = None):
        self._lock = threading.Lock()
        self._log_fn = log_fn or (lambda msg: None)
        self.state: str = "idle"  # idle | focus | short_break | long_break
        self.completed_today: int = 0
        self._today_date: datetime.date = datetime.date.today()
        self._timer: Optional[threading.Timer] = None
        self._phase_start_time: Optional[datetime.datetime] = None
        self._phase_duration_seconds: int = 0

    def _log(self, msg: str) -> None:
        try:
            self._log_fn(f"[Pomodoro] {msg}")
        except Exception:
            pass

    def _reset_if_new_day(self) -> None:
        today = datetime.date.today()
        if today != self._today_date:
            self._today_date = today
            self.completed_today = 0
            self._log("New day detected; pomodoro count reset.")

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _start_phase_timer(self, seconds: int) -> None:
        self._phase_start_time = datetime.datetime.now()
        self._phase_duration_seconds = seconds
        self._timer = threading.Timer(seconds, self._on_phase_complete)
        self._timer.daemon = True
        self._timer.start()

    def _apply_lights(self, preset: dict) -> None:
        """Set both Govee and Nanoleaf to the given preset. Never raises."""
        # Govee
        try:
            govee = preset["govee"]
            set_lights_style(
                state="on",
                color_hex=govee["color_hex"],
                color_temp_k=govee["color_temp_k"],
                brightness=govee["brightness"],
            )
            self._log(f"Govee set to {govee}")
        except (LightsClientError, Exception) as e:
            self._log(f"Govee light update failed (non-fatal): {e}")

        # Nanoleaf
        try:
            nanoleaf.turn_on()
            scene = preset.get("nanoleaf_scene")
            if scene:
                scenes = nanoleaf.get_scene_list()
                if scene in scenes:
                    nanoleaf.set_effect(scene)
                    self._log(f"Nanoleaf scene set to {scene}")
                else:
                    r, g, b = preset["nanoleaf_rgb"]
                    nanoleaf.set_color_rgb(r, g, b)
                    self._log(f"Nanoleaf scene '{scene}' not found; set static color instead")
            else:
                r, g, b = preset["nanoleaf_rgb"]
                nanoleaf.set_color_rgb(r, g, b)
                self._log(f"Nanoleaf set to RGB {preset['nanoleaf_rgb']}")
            brightness = preset.get("nanoleaf_brightness")
            if brightness is not None:
                nanoleaf.set_brightness(brightness)
        except Exception as e:
            self._log(f"Nanoleaf light update failed (non-fatal): {e}")

    def _on_phase_complete(self) -> None:
        """Called by threading.Timer when a phase finishes."""
        with self._lock:
            if self.state == "focus":
                self.completed_today += 1
                self._log(f"Focus session complete. Total today: {self.completed_today}")
                if self.completed_today % POMODOROS_BEFORE_LONG_BREAK == 0:
                    self.state = "long_break"
                    duration = LONG_BREAK_MINUTES * 60
                    preset = LIGHT_PRESETS["long_break"]
                    self._log("Starting long break.")
                else:
                    self.state = "short_break"
                    duration = SHORT_BREAK_MINUTES * 60
                    preset = LIGHT_PRESETS["short_break"]
                    self._log("Starting short break.")
                self._apply_lights(preset)
                self._start_phase_timer(duration)

            elif self.state in ("short_break", "long_break"):
                self._log("Break complete. Starting next focus session.")
                self.state = "focus"
                self._apply_lights(LIGHT_PRESETS["focus"])
                self._start_phase_timer(FOCUS_MINUTES * 60)

    def _build_status(self) -> dict[str, Any]:
        """Build the status dict. Must be called under lock."""
        self._reset_if_new_day()
        elapsed = 0
        remaining = 0
        if self._phase_start_time and self.state != "idle":
            elapsed = int((datetime.datetime.now() - self._phase_start_time).total_seconds())
            remaining = max(0, self._phase_duration_seconds - elapsed)
        return {
            "state": self.state,
            "completed_today": self.completed_today,
            "elapsed_seconds": elapsed,
            "remaining_seconds": remaining,
            "phase_duration_seconds": self._phase_duration_seconds,
        }

    # -- Public API ------------------------------------------------------------

    def start_focus(self) -> dict[str, Any]:
        """Start a focus session. If already focusing, return current status."""
        with self._lock:
            self._reset_if_new_day()
            if self.state == "focus":
                self._log("Already in focus session.")
                return self._build_status()
            self._cancel_timer()
            self.state = "focus"
            self._log("Starting focus session.")
            self._apply_lights(LIGHT_PRESETS["focus"])
            self._start_phase_timer(FOCUS_MINUTES * 60)
            return self._build_status()

    def stop(self) -> dict[str, Any]:
        """Stop the timer and return to idle."""
        with self._lock:
            self._cancel_timer()
            was = self.state
            self.state = "idle"
            self._phase_start_time = None
            self._phase_duration_seconds = 0
            self._log(f"Timer stopped (was {was}).")
            return self._build_status()

    def skip(self) -> dict[str, Any]:
        """Skip to the next phase immediately."""
        with self._lock:
            self._cancel_timer()
            if self.state == "focus":
                self.completed_today += 1
                self._log(f"Skipped focus. Total today: {self.completed_today}")
                if self.completed_today % POMODOROS_BEFORE_LONG_BREAK == 0:
                    self.state = "long_break"
                    duration = LONG_BREAK_MINUTES * 60
                    preset = LIGHT_PRESETS["long_break"]
                else:
                    self.state = "short_break"
                    duration = SHORT_BREAK_MINUTES * 60
                    preset = LIGHT_PRESETS["short_break"]
                self._apply_lights(preset)
                self._start_phase_timer(duration)
            elif self.state in ("short_break", "long_break"):
                self._log("Skipped break. Starting focus.")
                self.state = "focus"
                self._apply_lights(LIGHT_PRESETS["focus"])
                self._start_phase_timer(FOCUS_MINUTES * 60)
            else:
                self._log("Nothing to skip (idle).")
            return self._build_status()

    def get_status(self) -> dict[str, Any]:
        """Return current timer state."""
        with self._lock:
            return self._build_status()


# -- Singleton -----------------------------------------------------------------

_instance: Optional[PomodoroTimer] = None
_instance_lock = threading.Lock()


def get_timer(log_fn: Optional[Callable[[str], None]] = None) -> PomodoroTimer:
    """Return the global PomodoroTimer singleton."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = PomodoroTimer(log_fn=log_fn)
        return _instance
