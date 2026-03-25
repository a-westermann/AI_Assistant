"""
One-shot Spotify login for the PC running Galadrial.

Run from a normal terminal (not while holding a browser request to the API):

  python -m music.spotify_auth

This opens the browser, completes OAuth, and writes the token cache so
``/music/spotify/devices`` and Spotify play routes work without hanging.
"""

from __future__ import annotations

from .spotify_client import SpotifyNotConfiguredError, spotify_auth_interactive


def main() -> None:
    try:
        spotify_auth_interactive()
    except SpotifyNotConfiguredError as e:
        print(e)
        raise SystemExit(1) from e
    print("Spotify authorization saved. Retry GET /music/spotify/devices")


if __name__ == "__main__":
    main()
