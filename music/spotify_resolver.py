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
import random
import time
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

# Generic play <query> (treated as Spotify).
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
# Substring / voice-friendly (not anchored): "pause the playback please", "hey pause playback"
_RE_PAUSE_FLEX = re.compile(
    r"\b(?:pause|stop)\s+(?:the\s+)?(?:spotify\s+)?playback\b",
    re.IGNORECASE,
)
_RE_PAUSE_SPOTIFY_WORD = re.compile(
    r"\b(?:pause|stop)\s+spotify\b",
    re.IGNORECASE,
)

# Skip / next track — exact phrasing (after assistant-prefix strip); optional please at start or end.
_RE_SKIP_SONG = re.compile(
    r"^\s*(?:please\s+)?skip\s+song(?:\s+please)?\s*$",
    re.IGNORECASE,
)
_RE_NEXT_SONG = re.compile(
    r"^\s*(?:please\s+)?next\s+song(?:\s+please)?\s*$",
    re.IGNORECASE,
)


def _normalize_pause_phrase(text: str) -> str:
    """Strip assistant prefixes and trailing punctuation so pause intent still matches."""
    t = (text or "").strip()
    t = re.sub(r"^galadrial\s*[,:]\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^hey\s+galadrial\s*[,:]?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[\s.!?…]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass
class SpotifyResolution:
    ok: bool
    uri: str | None = None
    uris: list[str] | None = None
    kind: str = "track"  # track | playlist | album
    name: str | None = None
    artists: str | None = None
    search_query: str | None = None
    raw_intent: str | None = None
    playback_started: bool = False
    queued_count: int = 0
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
    t = _normalize_pause_phrase(text)
    if not t:
        return False
    if _RE_PAUSE.match(t) or _RE_SPOTIFY_PAUSE.match(t):
        return True
    if _RE_PAUSE_FLEX.search(t) or _RE_PAUSE_SPOTIFY_WORD.search(t):
        return True
    return False


def looks_like_spotify_skip_request(text: str) -> bool:
    """True for ``Skip song`` / ``Next song`` (case-insensitive, optional ``please `` prefix)."""
    if not text or not str(text).strip():
        return False
    t = _normalize_pause_phrase(text)
    if not t:
        return False
    return bool(_RE_SKIP_SONG.match(t) or _RE_NEXT_SONG.match(t))


def extract_spotify_play_intent(text: str) -> str | None:
    """Return the search phrase (without 'on Spotify' wrappers)."""
    t = (text or "").strip()

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
    """Backward-compatible: search string only (kind from :func:`refine_spotify_play_plan`)."""
    q, _k, _ts = refine_spotify_play_plan(intent_phrase)
    return q


def refine_spotify_play_plan(intent_phrase: str) -> tuple[str | None, str | None, str | None]:
    """
    Ask the local LLM for a Spotify search string, search kind, and track resolution mode.

    Soundtracks / scores should use ``album`` so we play the official release, not a
    loose track search full of fan covers and keyword spam.
    """
    prompt = (
        "The user asked to play music on Spotify. Their phrase is:\n"
        f"{intent_phrase!r}\n\n"
        "Reply with JSON only, no markdown:\n"
        '{"spotify_search_query": "best string to paste into Spotify search", '
        '"kind": "track|album|playlist|artist", '
        '"track_source": "track_search|artist_top_tracks"}\n\n'
        "Rules for kind:\n"
        "- album: soundtracks, film/game scores, OST, \"the X soundtrack\", named albums, "
        '"the Red album", "Dark Side of the Moon".\n'
        "- playlist: they said playlist, or a curated list by name/mood that is clearly playlist-shaped.\n"
        "- artist: user only names a band/artist (no specific song). Same as track with track_source "
        "artist_top_tracks; you may omit track_source when using artist.\n"
        "- track: a specific song title, one artist plus vibe, or vague genre/style (\"some jazz\", \"lo-fi beats\").\n"
        "Rules for track_source (when kind is track, or omitted for kind artist):\n"
        "- artist_top_tracks: user names a **band or artist** and did **not** ask for one specific song "
        '(e.g. "play The Mars Volta", "put on Radiohead"). Use Spotify\'s artist top-tracks for that '
        "artist (then the app shuffles). This avoids global track search always returning the same #1 hit.\n"
        "- track_search: a specific song, a genre/mood query (\"lo-fi beats\"), collaborations, or anything "
        "that is not \"just play this artist\".\n"
        "For film soundtracks, put composer + movie title in spotify_search_query when you know them "
        '(e.g. "Howard Shore The Two Towers").\n'
        "Keep spotify_search_query concise and specific so Spotify returns the right release first."
    )
    try:
        response = ask_lmstudio(prompt)
        raw = _lmstudio_text(response)
        obj = _parse_json_object(raw)
        if not obj:
            logger.warning("spotify_resolver: could not parse LLM JSON for Spotify play plan")
            return None, None, None
        q = obj.get("spotify_search_query")
        k = obj.get("kind")
        ts_raw = obj.get("track_source")
        kind_out: str | None = None
        from_artist_kind = False
        if isinstance(k, str):
            kl = k.strip().lower()
            if kl == "artist":
                kind_out = "track"
                from_artist_kind = True
            elif kl in ("track", "album", "playlist"):
                kind_out = kl
        track_source_out: str | None = None
        if isinstance(ts_raw, str):
            tl = ts_raw.strip().lower().replace(" ", "_").replace("-", "_")
            if tl in ("artist_top_tracks", "track_search"):
                track_source_out = tl
        if isinstance(q, str) and q.strip() and kind_out:
            if kind_out != "track":
                track_source_out = "track_search"
            elif from_artist_kind:
                track_source_out = "artist_top_tracks"
            elif track_source_out is None:
                track_source_out = "track_search"
            return q.strip(), kind_out, track_source_out
    except Exception as e:
        logger.warning("spotify_resolver: play plan LLM call failed: %s", e)
    return None, None, None


# Spotify allows up to 50 items per search page; one request keeps latency low.
_SPOTIFY_TRACK_SEARCH_LIMIT = 50
# Max tracks the LLM may choose for one playback round; shuffle applies only to this final list.
_FILTERED_TRACK_QUEUE_SIZE = 10


def _playback_target_n(queue_size: int, pool_len: int) -> int:
    """How many tracks to play: caller cap, global max, and available count."""
    return max(1, min(int(queue_size), _FILTERED_TRACK_QUEUE_SIZE, pool_len))


def _track_rows_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(candidates[:_SPOTIFY_TRACK_SEARCH_LIMIT]):
        if not isinstance(t, dict):
            continue
        uri = t.get("uri")
        if not isinstance(uri, str) or not uri:
            continue
        nm = (t.get("name") or "").strip()
        ar = sc.track_artists_label(t)
        rows.append({"i": i, "uri": uri, "artist": ar, "name": nm})
    return rows


def _is_narrow_score_intent(raw: str, search_query: str) -> bool:
    """Soundtrack/score-style requests: do not pad the pool with extra random tracks."""
    blob = f"{raw} {search_query}".lower()
    return any(
        k in blob
        for k in (
            "soundtrack",
            "soundtracks",
            "film score",
            "movie score",
            "original score",
            "score from",
            " ost",
            "ost ",
        )
    ) or re.search(r"\bost\b", blob) is not None


def _pad_indices_to_target(
    picked: list[int],
    valid_indices: set[int],
    target_n: int,
) -> list[int]:
    """If the model under-filled, add random unused indices so shuffle has enough variety."""
    out = list(picked)
    if len(out) >= target_n:
        return out[:target_n]
    pool = [i for i in valid_indices if i not in out]
    random.shuffle(pool)
    for idx in pool:
        out.append(idx)
        if len(out) >= target_n:
            break
    return out


def _llm_pick_track_indices(
    raw_intent: str,
    search_query: str,
    rows: list[dict[str, Any]],
    *,
    max_indices: int,
) -> list[int]:
    """Return candidate indices (into the original Spotify results list) in best-first order."""
    if not rows:
        return []
    if len(rows) == 1:
        return [rows[0]["i"]]

    lines = [f"{r['i']}. {r['artist']} — {r['name']}" for r in rows]
    prompt = (
        "You are choosing which Spotify search results match the user's request. Another step will **shuffle** "
        "your chosen tracks for playback, so **order in your JSON does not matter** — only **which** indices you include.\n\n"
        f"User request: {raw_intent!r}\n"
        f"Search query used: {search_query!r}\n\n"
        "Candidates (index = Spotify result index, artist — title):\n"
        + "\n".join(lines)
        + "\n\n"
        "Reply with JSON only, no markdown:\n"
        '{"indices": [3, 0, 7]}\n'
        "Use the index number exactly as shown before each line.\n"
        "Rules:\n"
        "- **Include** tracks that clearly fit the user's intent (correct artist, soundtrack, album, song, genre, or mood).\n"
        "- **Exclude** tracks that do not fit: unrelated keywords, fan uploads, wrong artist, joke/tribute, or wrong genre "
        "unless the user asked for those.\n"
        "- **Distinct tracks only** — never list the same index twice; never duplicate the same song.\n"
        f"- **Target {max_indices} indices** whenever enough distinct matching tracks exist in the list. "
        "Under-filling is wrong if the candidates clearly contain more matches. Only return fewer when fewer than "
        f"{max_indices} candidates actually match.\n"
        "- If the user asked for a **band or artist** (no specific song), include **as many distinct tracks by that artist "
        f"as you can, up to {max_indices}** — spread across albums/eras when possible, not one hit only.\n"
        "- If the user asked for a **genre or mood** (e.g. lo-fi), pick **diverse** fitting tracks, up to "
        f"{max_indices}.\n"
        f"- At most {max_indices} indices.\n"
        "- Use a one-element list **only** when exactly one candidate matches."
    )
    response = ask_lmstudio(prompt)
    raw = _lmstudio_text(response)
    obj = _parse_json_object(raw)
    if not obj:
        return []
    indices = obj.get("indices")
    if not isinstance(indices, list):
        return []
    out: list[int] = []
    valid = {r["i"] for r in rows}
    for x in indices:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi in valid and xi not in out:
            out.append(xi)
        if len(out) >= max_indices:
            break
    return out


def _ordered_tracks_from_indices(candidates: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    by_i = {i: t for i, t in enumerate(candidates) if isinstance(t, dict)}
    return [by_i[i] for i in indices if i in by_i]


def _shuffle_final_queue(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Last playback step: shuffle the list whose length is already final.

    Call only after the selection is fixed (e.g. LLM picked exactly the target count,
    or the pool was sliced to N). Do not cap or trim here.
    """
    pool = list(tracks)
    random.shuffle(pool)
    return pool


def _try_artist_top_tracks_candidates(sp: Any, search_query: str) -> list[dict[str, Any]] | None:
    """Resolve artist search → Spotify top tracks (avoids global track-search bias)."""
    artists = sc.search_artists(sp, search_query, limit=10)
    if not artists:
        return None
    sq_norm = search_query.strip().lower()
    artist_id: str | None = None
    for a in artists:
        nm = (a.get("name") or "").strip().lower()
        if nm == sq_norm:
            artist_id = a.get("id")
            break
    if artist_id is None:
        artist_id = artists[0].get("id")
    if not isinstance(artist_id, str):
        return None
    tracks = sc.artist_top_tracks(sp, artist_id)
    out = [t for t in tracks if isinstance(t, dict) and t.get("uri")]
    return out if out else None


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
    queue_size: int = 10,
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
    kind: str = "track"
    track_source: str = "track_search"
    if not skip_llm_refinement:
        rq, rk, rts = refine_spotify_play_plan(raw)
        if not rq or rk not in ("track", "album", "playlist"):
            return SpotifyResolution(
                ok=False,
                raw_intent=raw,
                search_query=raw,
                error=(
                    "Could not get a valid Spotify play plan from the language model "
                    "(need JSON with spotify_search_query and kind). Check LM Studio."
                ),
            )
        search_query = rq
        kind = rk
        track_source = rts or "track_search"
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
    uris: list[str] | None = None
    name: str | None = None
    artists: str | None = None

    try:
        if kind == "playlist":
            found = sc.search_playlists(sp, search_query, limit=20)
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
            pick = None
            ath_candidates: list[dict[str, Any]] | None = None
            if (
                not skip_llm_refinement
                and track_source == "artist_top_tracks"
                and kind == "track"
            ):
                ath_candidates = _try_artist_top_tracks_candidates(sp, search_query)
                if ath_candidates:
                    logger.info(
                        "spotify_resolver: artist_top_tracks path for %r (%d tracks)",
                        search_query,
                        len(ath_candidates),
                    )

            if ath_candidates:
                ath_n = _playback_target_n(queue_size, len(ath_candidates))
                ordered = _shuffle_final_queue(ath_candidates[:ath_n])
                pick = ordered[0]
                uri = pick.get("uri")
                name = pick.get("name")
                artists = sc.track_artists_label(pick)
                uris = [str(t.get("uri")) for t in ordered if isinstance(t.get("uri"), str)]
            else:
                found = sc.search_tracks(sp, search_query, limit=_SPOTIFY_TRACK_SEARCH_LIMIT)
                candidates = [t for t in found if isinstance(t, dict) and isinstance(t.get("uri"), str)]
                if candidates:
                    # Spotify returns the same ranking every time; shuffle so the LLM sees a fresh mix per request.
                    random.shuffle(candidates)
                    rows = _track_rows_from_candidates(candidates)
                    target_n = _playback_target_n(queue_size, len(rows))
                    if not skip_llm_refinement:
                        picked_idx = _llm_pick_track_indices(
                            raw,
                            search_query,
                            rows,
                            max_indices=target_n,
                        )
                        if not picked_idx:
                            return SpotifyResolution(
                                ok=False,
                                raw_intent=raw,
                                search_query=search_query,
                                kind=kind,
                                error=(
                                    "The language model did not return usable track indices for filtering. "
                                    "Check LM Studio."
                                ),
                            )
                        if not _is_narrow_score_intent(raw, search_query):
                            valid_idx = {r["i"] for r in rows}
                            picked_idx = _pad_indices_to_target(picked_idx, valid_idx, target_n)
                    else:
                        picked_idx = [r["i"] for r in rows[:target_n]]
                    ordered = _ordered_tracks_from_indices(candidates, picked_idx)
                    if not ordered:
                        return SpotifyResolution(
                            ok=False,
                            raw_intent=raw,
                            search_query=search_query,
                            kind=kind,
                            error="Track filtering produced no playable results.",
                        )
                    if len(ordered) > target_n:
                        ordered = ordered[:target_n]
                    ordered = _shuffle_final_queue(ordered)
                    pick = ordered[0]
                    uri = pick.get("uri")
                    name = pick.get("name")
                    artists = sc.track_artists_label(pick)
                    uris = [str(t.get("uri")) for t in ordered if isinstance(t.get("uri"), str)]
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
        uris=uris,
        search_query=search_query,
        raw_intent=raw,
        device_id=dev,
    )

    if not attempt_playback:
        return res

    try:
        sc.start_playback(
            sp,
            uri=uri,
            uris=uris if kind == "track" else None,
            kind=kind,
            device_id=dev,
        )
        res.playback_started = True
        if kind == "track" and isinstance(uris, list) and len(uris) > 1:
            res.queued_count = len(uris) - 1
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


def _resolve_pause_device_id(sp, explicit: str | None) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Pick the best device to pause. Prefer what Spotify says is *currently playing*,
    then SPOTIFY_DEVICE_ID, then is_active from /devices, then name match.
    """
    devices_list: list[dict[str, Any]] = []
    if explicit:
        return explicit, devices_list

    try:
        did = sc.get_active_playback_device_id(sp)
        if did:
            logger.info("pause_spotify_playback: device from current_playback: %r", did)
            try:
                devices_list = sc.list_connect_devices(sp)
            except Exception:
                pass
            return did, devices_list
    except Exception as e:
        logger.warning("pause_spotify_playback: current_playback device lookup failed: %s", e)

    try:
        devices_list = sc.list_connect_devices(sp)
    except Exception as e:
        logger.warning("pause_spotify_playback: list_connect_devices failed: %s", e)
        devices_list = []

    for d in devices_list:
        if d.get("is_active") and d.get("id"):
            logger.info("pause_spotify_playback: device from is_active in /devices: %r", d.get("id"))
            return str(d.get("id")), devices_list

    dev = sc.default_device_id()
    if dev:
        logger.info("pause_spotify_playback: device from SPOTIFY_DEVICE_ID")
        return dev, devices_list

    desired_name = (os.environ.get("SPOTIFY_DEVICE_NAME") or "Galadrial_Pi").strip()
    desired_norm = _norm_device_name(desired_name)
    for d in devices_list:
        nm_norm = _norm_device_name(str(d.get("name") or ""))
        if nm_norm and desired_norm and (desired_norm in nm_norm or nm_norm in desired_norm):
            did = d.get("id")
            if did:
                logger.info("pause_spotify_playback: device from name match %r", did)
                return str(did), devices_list

    return None, devices_list


def _playback_still_running(sp) -> bool:
    try:
        cp = sp.current_playback()
    except Exception:
        return False
    if not isinstance(cp, dict):
        return False
    return cp.get("is_playing") is True


def pause_spotify_playback(device_id: str | None = None) -> tuple[bool, str, str | None]:
    """
    Pause Spotify playback. Returns (ok, detail, used_device_id).

    Resolves the target device in order: explicit arg, then current session device,
    SPOTIFY_DEVICE_ID, is_active from /devices, then name match, then unqualified pause.
    Verifies is_playing clears when possible; retries once.
    """
    if not sc.spotify_credentials_configured():
        return False, "Spotify is not configured.", None
    try:
        sp = sc.get_spotify()
    except Exception as e:
        return False, str(e), None

    dev, _devices = _resolve_pause_device_id(sp, device_id)

    try:
        sc.pause_user_playback(sp, device_id=dev)
        time.sleep(0.45)
        if _playback_still_running(sp):
            logger.warning(
                "pause_spotify_playback: still playing after pause; retrying (device_id=%r)",
                dev,
            )
            sc.pause_user_playback(sp, device_id=dev)
            time.sleep(0.45)
            if _playback_still_running(sp):
                return (
                    False,
                    "Spotify still reports active playback after pause. "
                    "Try selecting Galadrial_Pi in the Spotify app once, or set SPOTIFY_DEVICE_ID to its id from /music/spotify/devices.",
                    dev,
                )
        return True, "", dev
    except Exception as e:
        return False, str(e), dev


def skip_spotify_track(device_id: str | None = None) -> tuple[bool, str, str | None]:
    """
    Skip to the next track (Spotify Connect). Uses the same device resolution order as pause.
    """
    if not sc.spotify_credentials_configured():
        return False, "Spotify is not configured.", None
    try:
        sp = sc.get_spotify()
    except Exception as e:
        return False, str(e), None

    dev, _devices = _resolve_pause_device_id(sp, device_id)

    try:
        sc.skip_to_next_track(sp, device_id=dev)
        return True, "", dev
    except Exception as e:
        return False, str(e), dev


def format_spotify_resolution_reply(res: SpotifyResolution) -> str:
    if not res.ok:
        return f"I couldn't play that on Spotify: {res.error or 'unknown error'}."
    base = f"I'll put on {res.raw_intent or 'that'} on Spotify."
    if res.playback_started:
        if res.queued_count > 0:
            return base + f" Started playback and queued {res.queued_count} more tracks."
        return base + " Playback started on your Connect device."
    if res.playback_error:
        return base + f" I found a match, but playback failed: {res.playback_error}"
    return base
