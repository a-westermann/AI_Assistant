"""
Music: YouTube ``Play …`` → ``video_id``; Spotify \"… on Spotify\" → search + Connect playback.
Imports are lazy so ``python -m music.play_resolver`` avoids import-order warnings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .play_resolver import (
        PlayResolution,
        extract_play_intent,
        format_play_resolution_reply,
        looks_like_play_music_request,
        refine_youtube_search_query,
        resolve_play_to_video_id,
        youtube_search_first_video,
        youtube_search_pick_video,
    )
    from .spotify_resolver import (
        SpotifyResolution,
        extract_spotify_play_intent,
        format_spotify_resolution_reply,
        looks_like_spotify_play_request,
        resolve_spotify_play,
    )

__all__ = [
    "PlayResolution",
    "extract_play_intent",
    "format_play_resolution_reply",
    "looks_like_play_music_request",
    "refine_youtube_search_query",
    "resolve_play_to_video_id",
    "youtube_search_first_video",
    "youtube_search_pick_video",
    "SpotifyResolution",
    "extract_spotify_play_intent",
    "format_spotify_resolution_reply",
    "looks_like_spotify_play_request",
    "resolve_spotify_play",
]

_EXPORT_MODULE: dict[str, str] = {
    **{name: "play_resolver" for name in (
        "PlayResolution",
        "extract_play_intent",
        "format_play_resolution_reply",
        "looks_like_play_music_request",
        "refine_youtube_search_query",
        "resolve_play_to_video_id",
        "youtube_search_first_video",
        "youtube_search_pick_video",
    )},
    **{name: "spotify_resolver" for name in (
        "SpotifyResolution",
        "extract_spotify_play_intent",
        "format_spotify_resolution_reply",
        "looks_like_spotify_play_request",
        "resolve_spotify_play",
    )},
}


def __getattr__(name: str):
    mod_name = _EXPORT_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(f"{__name__}.{mod_name}")
    return getattr(mod, name)
