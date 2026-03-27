"""Music: Spotify play intent resolution + Connect playback control."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spotify_resolver import (
        SpotifyResolution,
        extract_spotify_play_intent,
        looks_like_spotify_pause_request,
        format_spotify_resolution_reply,
        looks_like_spotify_play_request,
        looks_like_spotify_skip_request,
        pause_spotify_playback,
        resolve_spotify_play,
        skip_spotify_track,
    )

__all__ = [
    "SpotifyResolution",
    "extract_spotify_play_intent",
    "looks_like_spotify_pause_request",
    "format_spotify_resolution_reply",
    "looks_like_spotify_play_request",
    "looks_like_spotify_skip_request",
    "pause_spotify_playback",
    "resolve_spotify_play",
    "skip_spotify_track",
]

_EXPORT_MODULE: dict[str, str] = {
    **{name: "spotify_resolver" for name in (
        "SpotifyResolution",
        "extract_spotify_play_intent",
        "looks_like_spotify_pause_request",
        "format_spotify_resolution_reply",
        "looks_like_spotify_play_request",
        "looks_like_spotify_skip_request",
        "pause_spotify_playback",
        "resolve_spotify_play",
        "skip_spotify_track",
    )},
}


def __getattr__(name: str):
    mod_name = _EXPORT_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(f"{__name__}.{mod_name}")
    return getattr(mod, name)
