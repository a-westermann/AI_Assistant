import json
import os
from typing import Any, Dict

_MEMORY_PATH = os.path.join(os.path.dirname(__file__), "user_memory.json")


def _empty() -> Dict[str, Any]:
    return {"aliases": {}}


def load_memory() -> Dict[str, Any]:
    """Load global user memory from disk (simple JSON)."""
    if not os.path.isfile(_MEMORY_PATH):
        return _empty()
    try:
        with open(_MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty()
        data.setdefault("aliases", {})
        return data
    except Exception:
        return _empty()


def save_memory(mem: Dict[str, Any]) -> None:
    """Persist memory back to disk."""
    os.makedirs(os.path.dirname(_MEMORY_PATH), exist_ok=True)
    with open(_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)


_memory: Dict[str, Any] = load_memory()


def remember_alias(key: str, value: str) -> None:
    """
    Remember that a short phrase (key) should expand to some longer text (value),
    e.g. 'writing mode' -> 'dim orange nanoleaf and warm white govee at 35% brightness'.
    """
    key_norm = key.strip().lower()
    val = value.strip()
    if not key_norm or not val:
        return
    _memory.setdefault("aliases", {})
    _memory["aliases"][key_norm] = val
    save_memory(_memory)


def resolve_alias(phrase: str) -> str | None:
    """Return the stored expansion for a phrase, if any."""
    key_norm = (phrase or "").strip().lower()
    if not key_norm:
        return None
    aliases = _memory.get("aliases") or {}
    val = aliases.get(key_norm)
    return val if isinstance(val, str) and val.strip() else None

