"""
Detect phrases like \"Play some lo-fi beats\" or \"Play Polica\", optionally refine the search
via the local LLM, then resolve to a YouTube ``video_id`` (YouTube Data API v3).

Pass that id to your own player (e.g. HTTP call to a Raspberry Pi).

Requires ``YOUTUBE_API_KEY``. If LM Studio is unreachable, the text after \"play\" is used as
the search query.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import requests

from llm import ask_lmstudio

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# Title/description heuristics for “live in a room / session / concert” uploads (not perfect).
_LIVE_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\blive\s+(?:at|on|from|in)\b",
        r"\blive\s+@",
        r"\bfull\s+performance\b",
        r"\(live\)",
        r"\[live\]",
        r"\blive\s+session\b",
        r"\blive\s+recording\b",
        r"\bin\s+concert\b",
        r"\bconcert\b",
        r"\bfestival\b",
        r"\bon\s+stage\b",
        r"\bkexp\b",
        r"\btiny\s+desk\b",
        r"\bacoustic\s+session\b",
        r"\bworld\s+cafe\b",
        r"\b(?:radio|tv)\s+.*\blive\b",
    )
)

# Leading "play" (optional "please"), then the rest of the phrase.
_PLAY_INTENT_RE = re.compile(
    r"^\s*(?:please\s+)?play\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class PlayResolution:
    """Result of mapping a user play phrase to YouTube."""

    ok: bool
    video_id: str | None
    title: str | None
    search_query: str | None
    raw_intent: str | None
    error: str | None = None


def looks_like_play_music_request(text: str) -> bool:
    """True if text looks like a music play command (starts with Play …)."""
    if not text or not str(text).strip():
        return False
    m = _PLAY_INTENT_RE.match(text.strip())
    if not m:
        return False
    rest = (m.group(1) or "").strip()
    return bool(rest)


def extract_play_intent(text: str) -> str | None:
    """
    If this is a play request, return the phrase after 'play' (trimmed).
    Otherwise None.
    """
    if not text:
        return None
    m = _PLAY_INTENT_RE.match(text.strip())
    if not m:
        return None
    rest = (m.group(1) or "").strip()
    return rest or None


def _lmstudio_text(response: dict[str, Any]) -> str:
    return (response.get("output") or [{}])[0].get("content", "").strip()


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def refine_youtube_search_query(intent_phrase: str) -> str | None:
    """
    Ask the local LLM for a short YouTube search query (artist/title/mood keywords).
    Returns None if the model response could not be parsed.
    """
    intent_phrase = (intent_phrase or "").strip()
    if not intent_phrase:
        return None

    prompt = (
        "The user asked to play music. Their phrase (after the word play) is:\n"
        f'"""{intent_phrase}"""\n\n'
        "Produce a concise YouTube search query (a few words) that would find a good official "
        "or popular upload to play. Prefer studio recordings or official audio over live concerts "
        "or session performances. Prefer artist + song or genre + mood. No quotes in the query.\n"
        'Reply with JSON only, no markdown, no explanation:\n'
        '{"youtube_search_query": "..."}'
    )
    try:
        response = ask_lmstudio(prompt)
        raw = _lmstudio_text(response)
        obj = _parse_json_object(raw)
        if not obj:
            logger.warning("play_resolver: could not parse LLM JSON for search query")
            return None
        q = obj.get("youtube_search_query")
        if isinstance(q, str) and q.strip():
            return q.strip()
    except requests.ConnectionError as e:
        # LM Studio not running / wrong host — expected in many setups; fall back to raw intent.
        logger.info(
            "play_resolver: LM Studio unreachable (%s); using raw play intent for YouTube search",
            e,
        )
    except requests.Timeout as e:
        logger.info("play_resolver: LM Studio timeout (%s); using raw play intent", e)
    except requests.RequestException as e:
        logger.warning("play_resolver: LM Studio request failed (%s); using raw play intent", e)
    except Exception:
        logger.exception("play_resolver: unexpected error in refine_youtube_search_query")
    return None


def _snippet_suggests_live(title: str, description: str) -> bool:
    blob = f"{title or ''} {description or ''}"
    return any(p.search(blob) for p in _LIVE_TITLE_PATTERNS)


def _parse_search_items(items: list[Any]) -> list[tuple[str, str, str]]:
    """Return list of (video_id, title, description)."""
    out: list[tuple[str, str, str]] = []
    for it in items:
        vid = (it.get("id") or {}).get("videoId")
        snippet = it.get("snippet") or {}
        title = snippet.get("title") or ""
        desc = snippet.get("description") or ""
        if vid:
            out.append((str(vid), str(title), str(desc)))
    return out


def youtube_search_pick_video(
    query: str,
    api_key: str | None = None,
    timeout: float = 15.0,
    *,
    max_results: int = 20,
    exclude_live: bool = True,
    randomize: bool = True,
) -> tuple[str | None, str | None, str | None]:
    """
    YouTube Data API v3: fetch up to ``max_results`` videos, optionally drop live-looking hits,
    then pick one at random (or the first if ``randomize`` is False).

    Returns (video_id, title, error_message).
    """
    key = api_key or os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        return None, None, "YOUTUBE_API_KEY is not set"

    q = (query or "").strip()
    if not q:
        return None, None, "empty search query"

    n = max(1, min(int(max_results), 50))

    params = {
        "part": "snippet",
        "type": "video",
        "maxResults": str(n),
        "q": q,
        "key": key,
    }
    try:
        r = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return None, None, f"YouTube API request failed: {e}"

    err = data.get("error")
    if err:
        msg = err.get("message", str(err))
        return None, None, f"YouTube API error: {msg}"

    items = data.get("items") or []
    if not items:
        return None, None, "no videos found for that query"

    parsed = _parse_search_items(items)
    if not parsed:
        return None, None, "response missing videoId"

    if exclude_live:
        non_live = [t for t in parsed if not _snippet_suggests_live(t[1], t[2])]
        pool = non_live if non_live else parsed
        if non_live:
            logger.debug(
                "play_resolver: filtered %s live-looking of %s results",
                len(parsed) - len(non_live),
                len(parsed),
            )
    else:
        pool = parsed

    if randomize:
        choice = random.choice(pool)
    else:
        choice = pool[0]

    return choice[0], choice[1] or None, None


def youtube_search_first_video(
    query: str,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> tuple[str | None, str | None, str | None]:
    """
    YouTube Data API v3: first video only (no live filter, no random).

    For the assistant play flow, use :func:`youtube_search_pick_video` or
    :func:`resolve_play_to_video_id` instead.
    """
    return youtube_search_pick_video(
        query,
        api_key=api_key,
        timeout=timeout,
        max_results=1,
        exclude_live=False,
        randomize=False,
    )


def resolve_play_to_video_id(
    user_text: str,
    *,
    api_key: str | None = None,
    skip_llm_refinement: bool = False,
    max_search_results: int = 20,
    exclude_live: bool = True,
    randomize_pick: bool = True,
) -> PlayResolution:
    """
    Full pipeline: detect play intent → optional LLM search refinement → YouTube search → video id.

    If skip_llm_refinement is True, the text after \"play\" is sent directly to YouTube search.

    By default, fetches up to ``max_search_results`` hits, drops titles that look like live
    sessions/concerts, then picks one at random so repeated requests (e.g. \"Play Polica\")
    are not always the same upload.
    """
    raw = extract_play_intent(user_text or "")
    if not raw:
        return PlayResolution(
            ok=False,
            video_id=None,
            title=None,
            search_query=None,
            raw_intent=None,
            error="not a play music request",
        )

    if skip_llm_refinement:
        search_query = raw
    else:
        search_query = refine_youtube_search_query(raw)
        if not search_query:
            search_query = raw

    video_id, title, err = youtube_search_pick_video(
        search_query,
        api_key=api_key,
        max_results=max_search_results,
        exclude_live=exclude_live,
        randomize=randomize_pick,
    )
    if err:
        return PlayResolution(
            ok=False,
            video_id=None,
            title=None,
            search_query=search_query,
            raw_intent=raw,
            error=err,
        )

    return PlayResolution(
        ok=True,
        video_id=video_id,
        title=title,
        search_query=search_query,
        raw_intent=raw,
        error=None,
    )


def format_play_resolution_reply(res: PlayResolution) -> str:
    """Short user-facing text after resolving a \"Play …\" request."""
    if res.ok and res.video_id:
        label = res.title or res.video_id
        return f'Found: {label} (video id: {res.video_id}).'
    return f"I couldn't find a video for that: {res.error or 'unknown error'}."


def run_cli() -> None:
    """Used by ``python -m music`` and ``python -m music.play_resolver``."""
    import sys

    logging.basicConfig(level=logging.INFO)
    line = " ".join(sys.argv[1:]).strip() or "Play some lo-fi beats"
    print("Input:", line)
    print("looks_like_play:", looks_like_play_music_request(line))
    print("intent:", extract_play_intent(line))
    res = resolve_play_to_video_id(line)
    print("search_query:", res.search_query)
    print("video_id:", res.video_id)
    print("title:", res.title)
    print("error:", res.error)


if __name__ == "__main__":
    run_cli()
