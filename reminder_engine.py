import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional


class Reminder:
    """A single scheduled reminder."""

    def __init__(self, reminder_id: int, message: str, fire_at: datetime, minutes: float):
        self.id = reminder_id
        self.message = message
        self.fire_at = fire_at
        self.minutes = minutes
        self.created_at = datetime.now()
        self.fired = False
        self.cancelled = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "message": self.message,
            "fire_at": self.fire_at.strftime("%Y-%m-%d %H:%M:%S"),
            "minutes": self.minutes,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "fired": self.fired,
            "cancelled": self.cancelled,
            "remaining_seconds": max(0, (self.fire_at - datetime.now()).total_seconds()) if not self.fired else 0,
        }


class ReminderEngine:
    """
    Thread-safe reminder manager.  Keeps reminders in memory with
    background timer threads.  When a reminder fires it is moved to a
    "fired" queue that the assistant checks on the next interaction.
    Optionally flashes lights via a callback.
    """

    def __init__(
        self,
        log_fn: Optional[Callable[[str], None]] = None,
        on_fire: Optional[Callable[[Reminder], None]] = None,
    ):
        self.log = log_fn or (lambda msg: None)
        self.on_fire = on_fire
        self._lock = threading.Lock()
        self._next_id = 1
        self._active: Dict[int, Reminder] = {}
        self._timers: Dict[int, threading.Timer] = {}
        self._fired_queue: List[Reminder] = []  # unacknowledged fired reminders

    # ---- public API ---------------------------------------------------------

    def add(self, message: str, minutes: float) -> Reminder:
        """Schedule a new reminder that fires in `minutes` from now."""
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            fire_at = datetime.now() + timedelta(minutes=minutes)
            reminder = Reminder(rid, message, fire_at, minutes)
            self._active[rid] = reminder

            delay_seconds = max(0, minutes * 60)
            timer = threading.Timer(delay_seconds, self._fire, args=(rid,))
            timer.daemon = True
            timer.start()
            self._timers[rid] = timer

            self.log(f"Reminder #{rid} set for {minutes:.1f}min: {message!r}")
            return reminder

    def cancel(self, reminder_id: int) -> bool:
        """Cancel a pending reminder by ID.  Returns True if found and cancelled."""
        with self._lock:
            reminder = self._active.get(reminder_id)
            if reminder is None or reminder.fired or reminder.cancelled:
                return False
            reminder.cancelled = True
            timer = self._timers.pop(reminder_id, None)
            if timer is not None:
                timer.cancel()
            del self._active[reminder_id]
            self.log(f"Reminder #{reminder_id} cancelled.")
            return True

    def list_active(self) -> List[Dict[str, Any]]:
        """Return all pending (not yet fired, not cancelled) reminders as dicts."""
        with self._lock:
            return [
                r.to_dict()
                for r in sorted(self._active.values(), key=lambda r: r.fire_at)
                if not r.fired and not r.cancelled
            ]

    def list_fired(self) -> List[Dict[str, Any]]:
        """Return all fired reminders still in the queue (not yet acknowledged)."""
        with self._lock:
            return [r.to_dict() for r in self._fired_queue]

    def pop_fired(self) -> List[Reminder]:
        """Pop all fired reminders from the queue (acknowledges them)."""
        with self._lock:
            fired = list(self._fired_queue)
            self._fired_queue.clear()
            return fired

    def peek_fired(self) -> List[Reminder]:
        """Peek at fired reminders without removing them."""
        with self._lock:
            return list(self._fired_queue)

    # ---- internal -----------------------------------------------------------

    def _fire(self, reminder_id: int) -> None:
        """Called by the timer thread when a reminder goes off."""
        with self._lock:
            reminder = self._active.pop(reminder_id, None)
            self._timers.pop(reminder_id, None)
            if reminder is None or reminder.cancelled:
                return
            reminder.fired = True
            self._fired_queue.append(reminder)
            self.log(f"Reminder #{reminder_id} fired: {reminder.message!r}")

        # Flash lights (outside lock to avoid deadlocks with external APIs)
        if self.on_fire is not None:
            try:
                self.on_fire(reminder)
            except Exception as e:
                self.log(f"Reminder light flash failed (non-fatal): {e}")


def _flash_lights_notification(reminder: "Reminder") -> None:
    """
    Briefly flash Nanoleaf panels to Galadrial's gold (#f0b34a) for 2 seconds,
    then restore previous color.  Best-effort; failures are non-fatal.
    """
    try:
        from nanoleaf import nanoleaf as nl

        # Flash to gold
        nl.turn_on()
        nl.set_color_rgb(240, 179, 74)  # #f0b34a
        nl.set_brightness(100)
        time.sleep(2)
        # Dim back down to a gentle warm glow so we don't blind anyone
        nl.set_brightness(50)
        nl.set_color_rgb(255, 200, 100)
    except Exception:
        pass  # non-fatal; lights may be unreachable


# Module-level singleton (lazily created)
_engine: Optional[ReminderEngine] = None
_engine_lock = threading.Lock()


def get_reminder_engine(
    log_fn: Optional[Callable[[str], None]] = None,
) -> ReminderEngine:
    """Return (or create) the module-level singleton ReminderEngine."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ReminderEngine(
                log_fn=log_fn,
                on_fire=_flash_lights_notification,
            )
        return _engine