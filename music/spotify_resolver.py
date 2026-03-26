"""
Spotify play intent: phrases like \"Play chill metal on Spotify\" or \"Spotify play …\".
Search resolution and LLM query refinement run **on the PC**; playback is started via
Spotify's API to your Connect device (e.g. librespot on the Pi).

Requires Spotipy + OAuth env vars on the PC (see :mod:`music.spotify_client`).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
import os
from typing import Any

from llm import ask_lmstudio

from . import spotify_client as sc

logger = logging.getLogger(__name__)

# play <anything> on spotify
_RE_PLAY_ON_SPOTIFY = re.compile(
    r"^\s*(?:please\s+)?play\s+(.+?)\s+on\s+spotify\s*$",
    re.IGNORECASE | re.DOTALL,
)
# spotify: play … / spotify play …
_RE_SPOTIFY_PREFIX = re.compile(
    r"^\s*spotify\s*[:\s]+\s*play\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
# play spotify …
_RE_PLAY_SPOTIFY = re.compile(
    r"^\s*(?:please\s+)?play\s+spotify\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)

# Generic play <query> (treated as Spotify unless user explicitly mentions YouTube).
_RE_PLAY_GENERIC = re.compile(
    r"^\s*(?:please\s+)?play\s+(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Pause / stop playback trigger phrases
_RE_PAUSE = re.compile(
    r"^\s*(?:please\s+)?(?:pause|stop)\s+(?:the\s+)?(?:spotify\s+)?playback\s*$",
    re.IGNORECASE,
)
_RE_SPOTIFY_PAUSE = re.compile(
    r"^\s*spotify\s*[:\s]+(?:pause|stop)\s*$",
    re.IGNORECASE,
)


@dataclass
class SpotifyResolution:
    ok: bool
    uri: str | None = None
    kind: str = "track"  # track | playlist | album
    name: str | None = None
    artists: str | None = None
    search_query: str | None = None
    raw_intent: str | None = None
    playback_started: bool = False
    device_id: str | None = None
    error: str | None = None
    playback_error: str | None = None


def looks_like_spotify_play_request(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    return extract_spotify_play_intent(text) is not None


def looks_like_spotify_pause_request(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    t = str(text).strip()
    return bool(_RE_PAUSE.match(t) or _RE_SPOTIFY_PAUSE.match(t))


def extract_spotify_play_intent(text: str) -> str | None:
    """Return the search phrase (without 'on Spotify' wrappers)."""
    t = (text or "").strip()

    # If the user explicitly mentions YouTube, don't treat it as a Spotify request.
    # (This keeps the old YouTube "Play ..." flow intact.)
    t_lower = t.lower()
    if "youtube" in t_lower or "yt" in t_lower:
        return None

    for rx in (_RE_PLAY_ON_SPOTIFY, _RE_SPOTIFY_PREFIX, _RE_PLAY_SPOTIFY):
        m = rx.match(t)
        if m:
            inner = (m.group(1) or "").strip()
            return inner or None

    # If it's just "play <query>", default to Spotify query.
    m = _RE_PLAY_GENERIC.match(t)
    if m:
        inner = (m.group(1) or "").strip()
        return inner or None

    return None


def _lmstudio_text(response: dict[str, Any]) -> str:
    return (response.get("output") or [{}])[0].get("content", "").strip()


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    if "```" in raw:
        parts = raw.split("```")
        for block in parts:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
    return None


def refine_spotify_search_query(intent_phrase: str) -> str | None:
    """Ask local LLM for a concise Spotify search string, or None to fall back to raw intent."""
    prompt = (
        "The user asked to play audio on Spotify. Their phrase is:\n"
        f"{intent_phrase!r}\n\n"
        "Reply with JSON only, no markdown:\n"
        '{"spotify_search_query": "short search string for Spotify (artist, track, style, or playlist name)"}'
    )
    try:
        response = ask_lmstudio(prompt)
        raw = _lmstudio_text(response)
        obj = _parse_json_object(raw)
        if not obj:
            logger.warning("spotify_resolver: could not parse LLM JSON for Spotify search query")
            return None
        q = obj.get("spotify_search_query")
        if isinstance(q, str) and q.strip():
            return q.strip()
    except Exception as e:
        logger.info("spotify_resolver: LM Studio unavailable for Spotify query (%s); using raw intent", e)
    return None


def _infer_search_kind(intent_lower: str) -> str:
    if "playlist" in intent_lower:
        return "playlist"
    if re.search(r"\balbum\b", intent_lower):
        return "album"
    return "track"


def _norm_device_name(s: str | None) -> str:
    """Normalize device names so matching works across spaces/underscores/hyphens."""
    if not s:
        return ""
    x = s.lower().strip()
    # Remove non-alphanumerics (includes spaces, underscores, hyphens, etc.)
    x = re.sub(r"[^a-z0-9]+", "", x)
    return x


def resolve_spotify_play(
    user_text: str,
    *,
    skip_llm_refinement: bool = False,
    attempt_playback: bool = True,
    device_id: str | None = None,
) -> SpotifyResolution:
    """
    Search Spotify (track / playlist / album) and optionally start playback on Connect
    (``device_id`` or env ``SPOTIFY_DEVICE_ID``).
    """
    raw = extract_spotify_play_intent(user_text or "")
    if not raw:
        return SpotifyResolution(
            ok=False,
            error="not a Spotify play request",
            raw_intent=None,
        )

    if not sc.spotify_credentials_configured():
        return SpotifyResolution(
            ok=False,
            raw_intent=raw,
            error="Spotify is not configured (set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET).",
        )

    search_query = raw
    if not skip_llm_refinement:
        refined = refine_spotify_search_query(raw)
        if refined:
            search_query = refined

    kind = _infer_search_kind(raw.lower())
    dev = device_id if device_id is not None else sc.default_device_id()

    try:
        sp = sc.get_spotify()
    except Exception as e:
        logger.exception("spotify_resolver: could not create Spotify client")
        return SpotifyResolution(
            ok=False,
            raw_intent=raw,
            search_query=search_query,
            error=str(e),
        )

    uri: str | None = None
    name: str | None = None
    artists: str | None = None

    try:
        if kind == "playlist":
            found = sc.search_playlists(sp, search_query, limit=5)
            pick = found[0] if found else None
            if pick:
                uri = pick.get("uri")
                name = pick.get("name")
        elif kind == "album":
            found = sc.search_albums(sp, search_query, limit=5)
            pick = found[0] if found else None
            if pick:
                uri = pick.get("uri")
                name = pick.get("name")
                arts = pick.get("artists")
                if isinstance(arts, list) and arts:
                    artists = ", ".join(
                        a.get("name") for a in arts if isinstance(a, dict) and a.get("name")
                    )
        else:
            found = sc.search_tracks(sp, search_query, limit=10)
            pick = found[0] if found else None
            if pick:
                uri = pick.get("uri")
                name = pick.get("name")
                artists = sc.track_artists_label(pick)
    except Exception as e:
        logger.exception("spotify_resolver: Spotify search failed")
        return SpotifyResolution(
            ok=False,
            raw_intent=raw,
            search_query=search_query,
            error=f"Spotify search failed: {e}",
        )

    if not uri:
        return SpotifyResolution(
            ok=False,
            raw_intent=raw,
            search_query=search_query,
            kind=kind,
            error="No results on Spotify for that query.",
        )

    # If the caller didn't specify device_id, try matching a stable device name
    # (e.g. librespot --name "Galadrial_Pi"). This makes assistant playback land
    # on the Pi even if Spotify's "active device" changes.
    if dev is None:
        desired_name = (os.environ.get("SPOTIFY_DEVICE_NAME") or "Galadrial_Pi").strip()
        desired_norm = _norm_device_name(desired_name)
        try:
            devices = sc.list_connect_devices(sp)
            logger.info("spotify_resolver: matching device by name. desired=%r norm=%r devices=%s", desired_name, desired_norm, [d.get("name") for d in devices])
            for d in devices:
                nm = (d.get("name") or "").strip()
                nm_norm = _norm_device_name(nm)
                if nm_norm and desired_norm and (desired_norm in nm_norm or nm_norm in desired_norm):
                    dev = d.get("id") or None
                    logger.info("spotify_resolver: matched device %r (id=%r)", nm, dev)
                    break
        except Exception:
            dev = None

    res = SpotifyResolution(
        ok=True,
        uri=uri,
        kind=kind,
        name=name,
        artists=artists,
        search_query=search_query,
        raw_intent=raw,
        device_id=dev,
    )

    if not attempt_playback:
        return res

    try:
        sc.start_playback(sp, uri=uri, kind=kind, device_id=dev)
        res.playback_started = True
    except Exception as e:
        err = str(e)
        logger.warning("spotify_resolver: start_playback failed: %s", err)
        res.playback_error = err
        if not dev:
            res.playback_error = (
                f"{err} Hint: set SPOTIFY_DEVICE_ID to your Pi (librespot) device id, "
                "or call GET /music/spotify/devices to list devices."
            )

    return res


def pause_spotify_playback(device_id: str | None = None) -> tuple[bool, str, str | None]:
    """
    Pause Spotify playback. Returns (ok, detail, used_device_id).
    """
    if not sc.spotify_credentials_configured():
        return False, "Spotify is not configured.", None
    try:
        sp = sc.get_spotify()
    except Exception as e:
        return False, str(e), None

    dev = device_id if device_id is not None else sc.default_device_id()
    if dev is None:
        desired_name = (os.environ.get("SPOTIFY_DEVICE_NAME") or "Galadrial_Pi").strip()
        desired_norm = _norm_device_name(desired_name)
        try:
            devices = sc.list_connect_devices(sp)
            for d in devices:
                nm_norm = _norm_device_name(str(d.get("name") or ""))
                if nm_norm and desired_norm and (desired_norm in nm_norm or nm_norm in desired_norm):
                    dev = d.get("id") or None
                    break
        except Exception:
            dev = None

    try:
        if dev:
            sp.pause_playback(device_id=dev)
        else:
            sp.pause_playback()
        return True, "", dev
    except Exception as e:
        return False, str(e), dev


def format_spotify_resolution_reply(res: SpotifyResolution) -> str:
    if not res.ok:
        return f"I couldn't play that on Spotify: {res.error or 'unknown error'}."
    label = res.name or res.uri
    if res.artists and res.kind == "track":
        label = f"{res.name} — {res.artists}"
    base = f"Spotify: {label} ({res.uri})."
    if res.playback_started:
        return base + " Playback started on your Connect device."
    if res.playback_error:
        return base + f" Found the match but playback failed: {res.playback_error}"
    return base
