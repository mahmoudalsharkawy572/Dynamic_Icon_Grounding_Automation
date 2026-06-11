"""Project-wide logging: console + persistent file in debug/run.log."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"
_configured = False


def setup_logging(debug_dir: Path, level: int = logging.INFO) -> None:
    """Configure root logging exactly once (console + file handler)."""
    global _configured
    if _configured:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)

    file_handler = logging.FileHandler(debug_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
