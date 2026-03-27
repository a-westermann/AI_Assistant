"""
Saved shopping recipes (name + ingredient lines with quantities).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

from .shopping_list_store import merge_ingredients_into_shopping_list, normalize_name

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RECIPES_PATH = os.path.join(DATA_DIR, "shopping_recipes.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    _ensure_data_dir()
    fd, tmp_path = tempfile.mkstemp(prefix="recipes_", suffix=".json", dir=DATA_DIR)
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


def _load_raw() -> dict[str, Any]:
    if not os.path.isfile(RECIPES_PATH):
        return {"recipes": []}
    with open(RECIPES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"recipes": []}
    recipes = data.get("recipes")
    if not isinstance(recipes, list):
        return {"recipes": []}
    return {"recipes": recipes}


def _save_recipes(recipes: list[dict[str, Any]]) -> None:
    _atomic_write_json(RECIPES_PATH, {"recipes": recipes})


def _normalize_ingredient(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip().lower()
    if not name:
        return None
    n = normalize_name(name)
    q = raw.get("quantity", 1)
    try:
        q_int = int(q)
    except Exception:
        q_int = 1
    q_int = max(1, min(999, q_int))
    return {
        "id": str(raw.get("id") or uuid.uuid4()),
        "name": name,
        "normalized_name": n,
        "quantity": q_int,
    }


def list_recipes() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in _load_raw().get("recipes") or []:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "")
        title = str(r.get("name") or "").strip()
        if not rid or not title:
            continue
        ings: list[dict[str, Any]] = []
        for ing in r.get("ingredients") or []:
            ni = _normalize_ingredient(ing if isinstance(ing, dict) else {})
            if ni:
                ings.append(ni)
        out.append(
            {
                "id": rid,
                "name": title.lower(),
                "ingredients": ings,
                "created_at": str(r.get("created_at") or ""),
                "updated_at": str(r.get("updated_at") or ""),
            }
        )
    return out


def get_recipe(recipe_id: str) -> dict[str, Any]:
    for r in list_recipes():
        if r["id"] == recipe_id:
            return r
    raise KeyError("recipe not found")


def create_recipe(name: str, ingredients: list[dict[str, Any]]) -> dict[str, Any]:
    title = (name or "").strip().lower()
    if not title:
        raise ValueError("name must be non-empty")
    ings: list[dict[str, Any]] = []
    for raw in ingredients:
        ni = _normalize_ingredient(raw if isinstance(raw, dict) else {})
        if ni:
            ings.append(ni)
    if not ings:
        raise ValueError("at least one ingredient is required")
    now = _now_iso()
    recipe = {
        "id": str(uuid.uuid4()),
        "name": title,
        "ingredients": ings,
        "created_at": now,
        "updated_at": now,
    }
    recipes = [r for r in _load_raw().get("recipes") or [] if isinstance(r, dict)]
    recipes.append(recipe)
    _save_recipes(recipes)
    return get_recipe(recipe["id"])


def update_recipe(recipe_id: str, name: str | None, ingredients: list[dict[str, Any]] | None) -> dict[str, Any]:
    raw_list = [r for r in _load_raw().get("recipes") or [] if isinstance(r, dict)]
    idx = next((i for i, r in enumerate(raw_list) if str(r.get("id")) == str(recipe_id)), -1)
    if idx < 0:
        raise KeyError("recipe not found")
    item = dict(raw_list[idx])
    if name is not None:
        title = name.strip().lower()
        if not title:
            raise ValueError("name must be non-empty")
        item["name"] = title
    if ingredients is not None:
        ings: list[dict[str, Any]] = []
        for raw in ingredients:
            ni = _normalize_ingredient(raw if isinstance(raw, dict) else {})
            if ni:
                ings.append(ni)
        if not ings:
            raise ValueError("at least one ingredient is required")
        item["ingredients"] = ings
    item["updated_at"] = _now_iso()
    raw_list[idx] = item
    _save_recipes(raw_list)
    return get_recipe(recipe_id)


def delete_recipe(recipe_id: str) -> None:
    raw_list = [r for r in _load_raw().get("recipes") or [] if isinstance(r, dict)]
    new_list = [r for r in raw_list if str(r.get("id")) != str(recipe_id)]
    if len(new_list) == len(raw_list):
        raise KeyError("recipe not found")
    _save_recipes(new_list)


def apply_recipe_to_shopping_list(recipe_id: str) -> list[dict[str, Any]]:
    recipe = get_recipe(recipe_id)
    payload = [{"name": x["name"], "quantity": x["quantity"]} for x in recipe.get("ingredients") or []]
    return merge_ingredients_into_shopping_list(payload)
