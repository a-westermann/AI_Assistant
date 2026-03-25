"""
Shared shopping list persistence and sorting.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ITEMS_PATH = os.path.join(DATA_DIR, "shopping_list.json")
SORT_ORDER_PATH = os.path.join(DATA_DIR, "shopping_sort_order.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def _compact_for_overlap(s: str) -> str:
    return "".join(ch for ch in normalize_name(s) if ch.isalnum())


def _has_four_char_overlap(a: str, b: str) -> bool:
    aa = _compact_for_overlap(a)
    bb = _compact_for_overlap(b)
    if len(aa) < 4 or len(bb) < 4:
        return False
    short, long_ = (aa, bb) if len(aa) <= len(bb) else (bb, aa)
    shingles = {short[i : i + 4] for i in range(0, len(short) - 3)}
    return any(chunk in long_ for chunk in shingles)


def _best_fuzzy_reference_index(name: str, reference_order: list[str]) -> int | None:
    for i, ref_name in enumerate(reference_order):
        if _has_four_char_overlap(name, ref_name):
            return i
    return None


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    _ensure_data_dir()
    fd, tmp_path = tempfile.mkstemp(prefix="shopping_", suffix=".json", dir=DATA_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _load_items_raw() -> dict[str, Any]:
    if not os.path.isfile(ITEMS_PATH):
        return {"items": []}
    with open(ITEMS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"items": []}
    items = data.get("items")
    if not isinstance(items, list):
        return {"items": []}
    return {"items": items}


def _save_items(items: list[dict[str, Any]]) -> None:
    _atomic_write_json(ITEMS_PATH, {"items": items})


def _load_sort_order_raw() -> dict[str, Any]:
    if not os.path.isfile(SORT_ORDER_PATH):
        return {"reference_order": [], "updated_at": _now_iso()}
    with open(SORT_ORDER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"reference_order": [], "updated_at": _now_iso()}
    ref = data.get("reference_order")
    if not isinstance(ref, list):
        ref = []
    out = [normalize_name(str(x)) for x in ref if normalize_name(str(x))]
    return {"reference_order": out, "updated_at": data.get("updated_at") or _now_iso()}


def _save_sort_order(reference_order: list[str]) -> None:
    payload = {"reference_order": [normalize_name(x) for x in reference_order if normalize_name(x)], "updated_at": _now_iso()}
    _atomic_write_json(SORT_ORDER_PATH, payload)


def get_sort_order() -> list[str]:
    return list(_load_sort_order_raw().get("reference_order") or [])


def set_sort_order(reference_order: list[str]) -> list[str]:
    cleaned = [normalize_name(x) for x in reference_order if normalize_name(x)]
    _save_sort_order(cleaned)
    # Re-save list so order reflects latest reference.
    _save_items(_sorted_items(_load_items_raw().get("items") or [], cleaned))
    return cleaned


def _sorted_items(items: list[dict[str, Any]], reference_order: list[str] | None = None) -> list[dict[str, Any]]:
    ref = reference_order if reference_order is not None else get_sort_order()
    ref_index = {name: i for i, name in enumerate(ref)}

    def _key(item: dict[str, Any]) -> tuple:
        n = normalize_name(str(item.get("normalized_name") or item.get("name") or ""))
        if n in ref_index:
            return (0, ref_index[n], 0, str(item.get("created_at") or ""), str(item.get("name") or "").lower())
        fuzzy_idx = _best_fuzzy_reference_index(n, ref)
        if fuzzy_idx is not None:
            return (0, fuzzy_idx, 1, str(item.get("created_at") or ""), str(item.get("name") or "").lower())
        # Unknown items: keep stable insertion order (created_at), not alphabetical.
        return (1, 10**9, 0, str(item.get("created_at") or ""), str(item.get("name") or "").lower())

    return sorted(items, key=_key)


def get_items() -> list[dict[str, Any]]:
    raw = _load_items_raw().get("items") or []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out = dict(item)
        out["name"] = str(out.get("name") or "").strip().lower()
        out["normalized_name"] = normalize_name(str(out.get("normalized_name") or out.get("name") or ""))
        q = out.get("quantity", 1)
        try:
            q_int = int(q)
        except Exception:
            q_int = 1
        out["quantity"] = max(1, min(999, q_int))
        normalized.append(out)
    return _sorted_items(normalized)


def add_item(name: str) -> dict[str, Any]:
    display = (name or "").strip().lower()
    if not display:
        raise ValueError("name must be non-empty")
    normalized = normalize_name(display)
    items = _load_items_raw().get("items") or []
    if any(normalize_name(str(i.get("normalized_name") or i.get("name") or "")) == normalized for i in items):
        raise FileExistsError("duplicate")
    now = _now_iso()
    item = {
        "id": str(uuid.uuid4()),
        "name": display,
        "normalized_name": normalized,
        "quantity": 1,
        "checked": False,
        "created_at": now,
        "updated_at": now,
    }
    items.append(item)
    _save_items(_sorted_items(items))
    return item


def update_item(
    item_id: str,
    *,
    name: str | None = None,
    checked: bool | None = None,
    quantity: int | None = None,
) -> dict[str, Any]:
    items = _load_items_raw().get("items") or []
    idx = next((i for i, it in enumerate(items) if str(it.get("id")) == str(item_id)), -1)
    if idx < 0:
        raise KeyError("item not found")
    item = dict(items[idx])
    if name is not None:
        display = name.strip().lower()
        if not display:
            raise ValueError("name must be non-empty")
        normalized = normalize_name(display)
        for j, other in enumerate(items):
            if j == idx:
                continue
            n = normalize_name(str(other.get("normalized_name") or other.get("name") or ""))
            if n == normalized:
                raise FileExistsError("duplicate")
        item["name"] = display
        item["normalized_name"] = normalized
    if checked is not None:
        item["checked"] = bool(checked)
    if quantity is not None:
        item["quantity"] = max(1, min(999, int(quantity)))
    item["updated_at"] = _now_iso()
    items[idx] = item
    _save_items(_sorted_items(items))
    return item


def delete_item(item_id: str) -> None:
    items = _load_items_raw().get("items") or []
    new_items = [it for it in items if str(it.get("id")) != str(item_id)]
    if len(new_items) == len(items):
        raise KeyError("item not found")
    _save_items(_sorted_items(new_items))


def replace_all(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    now = _now_iso()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip().lower()
        if not name:
            continue
        n = normalize_name(name)
        if n in dedup:
            continue
        item = {
            "id": str(raw.get("id") or uuid.uuid4()),
            "name": name,
            "normalized_name": n,
            "quantity": max(1, min(999, int(raw.get("quantity", 1)))),
            "checked": bool(raw.get("checked", False)),
            "created_at": str(raw.get("created_at") or now),
            "updated_at": str(raw.get("updated_at") or now),
        }
        dedup[n] = item
        out.append(item)
    sorted_out = _sorted_items(out)
    _save_items(sorted_out)
    return sorted_out
