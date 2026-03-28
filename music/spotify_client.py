"""
Spotify Web API via Spotipy (user OAuth). **Search and playback control both run on the PC**
(Galadrial / ``api_server``); the Raspberry Pi is only the Spotify **Connect** speaker
(librespot), not an HTTP proxy.

Environment:
  SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET — app credentials from Spotify Developer Dashboard
  SPOTIFY_REDIRECT_URI — default http://127.0.0.1:8888/callback (must match app settings)
  SPOTIFY_TOKEN_CACHE — path to token cache file (default: .cache-spotify in repo root)
  SPOTIFY_DEVICE_ID — optional default Connect device id (your Pi / librespot). If unset,
    Spotify uses the active device, which may not be the Pi.
  SPOTIFY_SKIP_QUEUE_CLEAR — if ``1``/``true``, do not drain the Connect queue before
    starting new playback (tracks, album, or playlist). Not recommended: old queue items
    can still play after the current context ends.
  SPOTIFY_USE_BULK_URIS_PLAYBACK — if ``1``/``true``, send all track URIs in one
    ``start_playback`` call (can confuse librespot; default is first track + ``add_to_queue``
    for the rest, after ``clear_playback_queue``).

Interactive login (once): run ``python -m music.spotify_auth`` from a terminal on the PC.
The API does not start OAuth during HTTP requests (that would hang the browser / worker).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_REDIRECT = "http://127.0.0.1:8888/callback"
_DEFAULT_SCOPE = (
    "user-modify-playback-state user-read-playback-state user-read-currently-playing"
)

_spotify: Any = None


class SpotifyNotConfiguredError(Exception):
    """Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET."""


class SpotifyAuthRequiredError(Exception):
    """No valid token on disk; run interactive auth once (see spotify_auth.py)."""


def spotify_credentials_configured() -> bool:
    cid = (os.environ.get("SPOTIFY_CLIENT_ID") or "").strip()
    sec = (os.environ.get("SPOTIFY_CLIENT_SECRET") or "").strip()
    return bool(cid and sec)


def default_device_id() -> str | None:
    d = (os.environ.get("SPOTIFY_DEVICE_ID") or "").strip()
    return d or None


def _repo_root_cache_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, ".cache-spotify")


def _open_browser_for_oauth() -> bool:
    """True only when explicitly enabled (never default on: would block FastAPI / hang the browser tab)."""
    return os.environ.get("SPOTIFY_OPEN_BROWSER", "").strip().lower() in ("1", "true", "yes")


def _auth_manager(*, open_browser: bool | None = None):
    if not spotify_credentials_configured():
        raise SpotifyNotConfiguredError(
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (see music/spotify_client.py docstring)."
        )
    try:
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError as e:
        raise RuntimeError(
            "spotipy is not installed. Add it with: pip install spotipy"
        ) from e

    cache = (os.environ.get("SPOTIFY_TOKEN_CACHE") or "").strip() or _repo_root_cache_path()
    redirect = (os.environ.get("SPOTIFY_REDIRECT_URI") or "").strip() or _DEFAULT_REDIRECT
    scope = (os.environ.get("SPOTIFY_SCOPE") or "").strip() or _DEFAULT_SCOPE
    ob = _open_browser_for_oauth() if open_browser is None else open_browser
    return SpotifyOAuth(
        client_id=os.environ["SPOTIFY_CLIENT_ID"].strip(),
        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"].strip(),
        redirect_uri=redirect,
        scope=scope,
        cache_path=cache,
        open_browser=ob,
    )


def spotify_has_cached_token() -> bool:
    """True if a token file exists and Spotipy reports a usable cached token (no network)."""
    if not spotify_credentials_configured():
        return False
    try:
        am = _auth_manager(open_browser=False)
        tok = am.get_cached_token()
        return bool(tok and tok.get("access_token"))
    except Exception:
        return False


def get_spotify():
    """
    Singleton Spotipy client (lazy). Does **not** start interactive OAuth from the API server
    (that would block HTTP handlers). Run ``python -m music.spotify_auth`` once to create the cache.
    """
    global _spotify
    if _spotify is None:
        if not spotify_has_cached_token():
            cache = (os.environ.get("SPOTIFY_TOKEN_CACHE") or "").strip() or _repo_root_cache_path()
            raise SpotifyAuthRequiredError(
                "Spotify is not logged in yet. From a desktop terminal on this PC run:\n"
                f"  python -m music.spotify_auth\n"
                f"Then retry. (Token file: {cache})"
            )
        import spotipy

        _spotify = spotipy.Spotify(auth_manager=_auth_manager(open_browser=False), requests_timeout=15)
    return _spotify


def reset_spotify_client() -> None:
    """Clear singleton after interactive auth (so the next call loads fresh credentials)."""
    global _spotify
    _spotify = None


def spotify_auth_interactive() -> None:
    """
    Open browser OAuth flow and write token cache. Run once from a normal terminal (not via uvicorn).
    """
    import spotipy

    if not spotify_credentials_configured():
        raise SpotifyNotConfiguredError(
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET before running spotify_auth."
        )
    reset_spotify_client()
    am = _auth_manager(open_browser=True)
    sp = spotipy.Spotify(auth_manager=am, requests_timeout=30)
    sp.current_user()
    reset_spotify_client()


def get_active_playback_device_id(sp=None) -> str | None:
    """
    Return the device id from GET /v1/me/player when there is an active playback session.

    This often works for librespot even when GET /v1/me/player/devices does not list the Pi,
    so pause/transfer can target the correct Connect device.
    """
    sp = sp or get_spotify()
    try:
        cp = sp.current_playback()
    except Exception:
        return None
    if not isinstance(cp, dict):
        return None
    dev = cp.get("device")
    if isinstance(dev, dict):
        did = dev.get("id")
        if isinstance(did, str) and did.strip():
            return did.strip()
    return None


def list_connect_devices(sp=None) -> list[dict[str, Any]]:
    """Devices visible to Spotify Connect (find your Pi / librespot ``id`` for SPOTIFY_DEVICE_ID)."""
    sp = sp or get_spotify()
    data = sp.devices()
    devs = data.get("devices") if isinstance(data, dict) else None
    if not isinstance(devs, list):
        return []
    out: list[dict[str, Any]] = []
    for d in devs:
        if isinstance(d, dict):
            out.append(
                {
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "is_active": d.get("is_active"),
                    "is_restricted": d.get("is_restricted"),
                    "type": d.get("type"),
                }
            )
    return out


def search_tracks(sp, q: str, *, limit: int = 10) -> list[dict[str, Any]]:
    r = sp.search(q=q, type="track", limit=limit)
    tracks = (r.get("tracks") or {}).get("items")
    return [t for t in tracks if isinstance(t, dict)] if isinstance(tracks, list) else []


def search_playlists(sp, q: str, *, limit: int = 5) -> list[dict[str, Any]]:
    r = sp.search(q=q, type="playlist", limit=limit)
    pl = (r.get("playlists") or {}).get("items")
    return [p for p in pl if isinstance(p, dict)] if isinstance(pl, list) else []


def search_albums(sp, q: str, *, limit: int = 5) -> list[dict[str, Any]]:
    r = sp.search(q=q, type="album", limit=limit)
    al = (r.get("albums") or {}).get("items")
    return [a for a in al if isinstance(a, dict)] if isinstance(al, list) else []


def search_artists(sp, q: str, *, limit: int = 10) -> list[dict[str, Any]]:
    r = sp.search(q=q, type="artist", limit=limit)
    items = (r.get("artists") or {}).get("items")
    return [a for a in items if isinstance(a, dict)] if isinstance(items, list) else []


def artist_top_tracks(sp, artist_id: str, *, country: str | None = None) -> list[dict[str, Any]]:
    """
    Spotify returns up to 10 top tracks for the artist (market from ``SPOTIFY_MARKET`` or ``US``).
    """
    c = (country or os.environ.get("SPOTIFY_MARKET") or "US").strip()
    r = sp.artist_top_tracks(artist_id, country=c)
    tracks = r.get("tracks")
    return [t for t in tracks if isinstance(t, dict)] if isinstance(tracks, list) else []


def track_artists_label(track: dict) -> str:
    arts = track.get("artists")
    if not isinstance(arts, list):
        return ""
    names = [a.get("name") for a in arts if isinstance(a, dict) and a.get("name")]
    return ", ".join(names)


def clear_playback_queue(
    sp,
    *,
    device_id: str | None = None,
    max_skips: int = 64,
) -> int:
    """
    Spotify has no API to clear the user's playback queue. Approximate a reset by pausing,
    then calling ``next_track`` until ``GET /me/player/queue`` reports an empty ``queue``
    list (capped at ``max_skips``). Call after ``transfer_playback`` so the target device
    matches. Some Connect targets may play a short burst when advancing.
    """
    try:
        sp.pause_playback(device_id=device_id)
    except Exception as e:
        logger.debug("clear_playback_queue: pause failed (idle session is ok): %s", e)
    time.sleep(0.28)
    skips = 0
    for _ in range(max(1, int(max_skips))):
        try:
            data = sp.queue()
        except Exception as e:
            logger.warning("clear_playback_queue: queue() failed: %s", e)
            break
        if not isinstance(data, dict):
            break
        upcoming = data.get("queue")
        if not isinstance(upcoming, list) or len(upcoming) == 0:
            break
        try:
            sp.next_track(device_id=device_id)
        except Exception as e:
            logger.warning("clear_playback_queue: next_track failed: %s", e)
            break
        skips += 1
        time.sleep(0.14)
    try:
        tail = sp.queue()
        up = tail.get("queue") if isinstance(tail, dict) else None
        if isinstance(up, list) and len(up) > 0:
            logger.warning(
                "clear_playback_queue: %d item(s) still in queue after drain (cap=%s)",
                len(up),
                max_skips,
            )
    except Exception:
        pass
    try:
        sp.pause_playback(device_id=device_id)
    except Exception:
        pass
    time.sleep(0.12)
    if skips:
        logger.info("spotify_client: drained %d queued track(s) before new playback", skips)
    return skips


def start_playback(
    sp,
    *,
    uri: str | None = None,
    uris: list[str] | None = None,
    kind: str,
    device_id: str | None = None,
) -> None:
    """
    ``kind`` is ``track`` | ``playlist`` | ``album``.

    Unless ``SPOTIFY_SKIP_QUEUE_CLEAR`` is set, always runs ``clear_playback_queue`` first
    so album/playlist playback does not inherit stale queued tracks from earlier sessions.
    """
    kwargs: dict[str, Any] = {}
    if device_id:
        # Force the target Connect device active first. Without this, Spotify may keep
        # playback on the last active client unless the user pre-selects the Pi manually.
        try:
            sp.transfer_playback(device_id=device_id, force_play=False)
            # Give Spotify a brief moment to apply the transfer before start_playback.
            time.sleep(0.25)
        except Exception as e:
            logger.warning("spotify_client: transfer_playback failed for device %r: %s", device_id, e)
        kwargs["device_id"] = device_id

    skip_clear = (os.environ.get("SPOTIFY_SKIP_QUEUE_CLEAR") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    dev = kwargs.get("device_id")
    if not skip_clear:
        # Any new play (tracks, album, or playlist) should not inherit yesterday's queue.
        clear_playback_queue(sp, device_id=dev)

    if kind == "track":
        bulk = (os.environ.get("SPOTIFY_USE_BULK_URIS_PLAYBACK") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if isinstance(uris, list) and uris:
            if len(uris) > 1 and bulk:
                sp.start_playback(uris=uris, **kwargs)
            elif len(uris) > 1:
                # Librespot often mishandles multiple URIs in one call; add_to_queue appends
                # to the *end* of the queue — we drain the queue above so "next" is only our list.
                sp.start_playback(uris=[uris[0]], **kwargs)
                time.sleep(0.2)
                for quri in uris[1:]:
                    if not isinstance(quri, str) or not quri:
                        continue
                    try:
                        sp.add_to_queue(quri, device_id=dev)
                    except Exception as e:
                        logger.warning("spotify_client: add_to_queue failed for %r: %s", quri, e)
                    time.sleep(0.05)
            else:
                sp.start_playback(uris=uris, **kwargs)
        elif isinstance(uri, str) and uri:
            sp.start_playback(uris=[uri], **kwargs)
        else:
            raise ValueError("track playback requires uri or uris")
    else:
        if not isinstance(uri, str) or not uri:
            raise ValueError("context playback requires uri")
        sp.start_playback(context_uri=uri, **kwargs)


def skip_to_next_track(sp, *, device_id: str | None = None) -> None:
    """
    Skip to the next track on the active (or specified) Connect device.
    Uses the same transfer-then-command pattern as :func:`pause_user_playback` for librespot.
    """
    if device_id:
        try:
            sp.transfer_playback(device_id=device_id, force_play=False)
            time.sleep(0.25)
        except Exception as e:
            logger.warning(
                "spotify_client: transfer_playback before skip failed for %r: %s",
                device_id,
                e,
            )
        sp.next_track(device_id=device_id)
    else:
        sp.next_track()


def pause_user_playback(sp, *, device_id: str | None = None) -> None:
    """
    Pause playback on a Connect device the same way we *start* playback: transfer the
    active session to that device first, then pause the active player.

    Calling ``pause_playback(device_id=...)`` alone is unreliable for some Connect
    targets (e.g. librespot): Spotify may accept the HTTP call but not route SPIRC
    pause to the speaker. Priming with ``transfer_playback(..., force_play=False)``
    matches our ``start_playback`` path and fixes that.
    """
    if device_id:
        try:
            sp.transfer_playback(device_id=device_id, force_play=False)
            time.sleep(0.35)
        except Exception as e:
            logger.warning(
                "spotify_client: transfer_playback before pause failed for %r: %s",
                device_id,
                e,
            )
        sp.pause_playback()
    else:
        sp.pause_playback()
