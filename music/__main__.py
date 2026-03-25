"""
CLI: ``python -m music "Play Polica"`` (same as ``python -m music.play_resolver``).
"""

from __future__ import annotations

from .play_resolver import run_cli

if __name__ == "__main__":
    run_cli()
