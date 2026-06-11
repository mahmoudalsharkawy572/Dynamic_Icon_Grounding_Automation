"""Entry point: ScreenSeekeR visual grounding + Notepad automation task.

Usage:
    python main.py                       # full task (locate + 10 posts)
    python main.py --locate-only         # just locate & annotate, no clicks
    python main.py --instruction "the Recycle Bin icon on the desktop"
    python main.py --posts 3
    python main.py --no-fallback         # disable the OS-launch fallback
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root into os.environ

from src.api_client import GeminiClient
from src.config import ConfigError, Settings
from src.executor import Executor
from src.grounder import Grounder
from src.logger import get_logger, setup_logging
from src.planner import Planner
from src.screenseeker import ScreenSeekeR
from src.screenshot import ScreenCapturer
from src.verifier import Verifier

log = get_logger("main")

POSTS_URL = "https://dummyjson.com/posts"
DEFAULT_INSTRUCTION = "the Notepad shortcut icon on the desktop"


# ----------------------------------------------------------------- task data

def fetch_posts(settings: Settings, limit: int) -> list[dict[str, Any]]:
    """Download posts with retries; raises after MAX_RETRIES (e.g. offline)."""
    last_exc: Exception | None = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            resp = requests.get(
                POSTS_URL,
                params={"limit": limit},  # DummyJSON supports server-side limiting
                timeout=settings.request_timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
            posts = payload.get("posts") if isinstance(payload, dict) else None
            if not isinstance(posts, list):
                raise ValueError(f"Unexpected payload structure: {type(payload)}")
            return posts[:limit]
        except Exception as exc:
            last_exc = exc
            log.warning("Fetching posts failed (attempt %d/%d): %s",
                        attempt, settings.max_retries, exc)
            time.sleep(settings.retry_delay_s)
    raise RuntimeError(f"Could not download posts after retries: {last_exc}")


def project_dir() -> Path:
    target = Path.home() / "Desktop" / "tjm-project"
    target.mkdir(parents=True, exist_ok=True)
    return target


# --------------------------------------------------------- desktop / notepad

def show_desktop(executor: Executor) -> None:
    """Minimize all windows so desktop icons are visible for grounding.

    Without this, the screenshot captures whatever window is in the
    foreground (e.g. the IDE running this script) and the icon can never
    be found.
    """
    executor.hotkey("win", "d")
    time.sleep(1.0)


def launch_notepad_via_os() -> bool:
    """Fallback: launch Notepad directly through the OS.

    Used ONLY when the ScreenSeekeR visual search exhausts all retries.
    This bypasses visual grounding, so it is logged loudly as a fallback.
    """
    log.warning("FALLBACK: launching Notepad via OS (visual grounding failed)")
    try:
        subprocess.Popen(["notepad.exe"])
        return True
    except (OSError, FileNotFoundError) as exc:
        log.error("OS fallback failed to start Notepad: %s", exc)
        return False


def notepad_window_open() -> bool:
    """Check for a Notepad window by title (Windows); falls back to False."""
    try:
        import pygetwindow as gw  # bundled with pyautogui on Windows
        return any("notepad" in (t or "").lower() for t in gw.getAllTitles())
    except Exception:
        return False


def verify_notepad_launched(
    settings: Settings, capturer: ScreenCapturer, verifier: Verifier
) -> bool:
    """Window-title check first; visual LLM check as a cross-platform fallback."""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if notepad_window_open():
            log.info("Notepad window detected via window title")
            return True
        time.sleep(0.5)
    log.info("Window-title check inconclusive; falling back to visual verification")
    shot = capturer.capture_primary()
    result = verifier.verify("an open Notepad text editor window", shot)
    return result.correct


def close_notepad(executor: Executor) -> None:
    """Close Notepad safely.

    NEVER sends Alt+F4 blindly: if Notepad is not the focused window,
    Alt+F4 lands on the desktop and opens the Windows SHUTDOWN dialog
    (and a subsequent Enter would confirm it). We therefore:
      1. Find the actual Notepad window and activate it first.
      2. Only then send Alt+F4.
      3. Answer any 'save changes?' prompt with 'n' (Don't save) — the file
         was already saved via Save As. Never press Enter blindly.
      4. On any failure, press Escape to dismiss whatever dialog appeared.
    """
    try:
        import pygetwindow as gw
        windows = [w for w in gw.getAllWindows()
                   if "notepad" in (w.title or "").lower()]
        if not windows:
            log.info("Notepad already closed")
            return
        win = windows[0]
        win.activate()  # guarantee keyboard focus is on Notepad
        time.sleep(0.5)
        executor.hotkey("alt", "f4")
        time.sleep(1.0)
        # If a 'save changes?' prompt kept it open, decline (Win11: Don't save).
        if any("notepad" in (w.title or "").lower() for w in gw.getAllWindows()):
            executor.press("n")
            time.sleep(0.5)
    except Exception as exc:
        log.warning("Window-targeted close failed (%s); pressing Escape to "
                    "dismiss any system dialog", exc)
        executor.press("esc")
        time.sleep(0.5)


def write_post_in_notepad(
    executor: Executor, post: dict[str, Any], save_path: Path
) -> None:
    """Type the post into the focused Notepad window and save it to save_path."""
    content = f"Title: {post.get('title', '')}\n\n{post.get('body', '')}"
    time.sleep(1.0)
    executor.paste_text(content)
    time.sleep(0.5)

    # Pre-delete target to avoid the 'Confirm Save As' overwrite dialog.
    if save_path.exists():
        save_path.unlink()

    executor.hotkey("ctrl", "shift", "s")  # Save As (Win11 Notepad)
    time.sleep(1.2)
    executor.paste_text(str(save_path))
    time.sleep(0.5)
    executor.press("enter")
    time.sleep(1.2)

    close_notepad(executor)

    # Park the mouse in a neutral corner so no tooltip is showing
    # in the next iteration's screenshot.
    executor.move_to(5, 5)


# ------------------------------------------------------------------ pipeline

def build_pipeline(settings: Settings) -> tuple[ScreenSeekeR, Executor, ScreenCapturer, Verifier]:
    client = GeminiClient(settings)
    capturer = ScreenCapturer()
    planner = Planner(client, settings)
    grounder = Grounder(client, settings)
    verifier = Verifier(client, settings)
    seeker = ScreenSeekeR(settings, capturer, planner, grounder, verifier)
    return seeker, Executor(settings), capturer, verifier


def run_task(
    settings: Settings,
    instruction: str,
    posts_count: int,
    locate_only: bool,
    allow_fallback: bool,
) -> int:
    seeker, executor, capturer, verifier = build_pipeline(settings)

    if locate_only:
        show_desktop(executor)
        result = seeker.locate(instruction)
        if not result.found:
            log.error("Target not found: %r", instruction)
            return 1
        log.info("FOUND %r at (%d,%d), depth=%d, region=%r",
                 instruction, result.screen_x, result.screen_y,
                 result.depth, result.region_description)
        return 0

    posts = fetch_posts(settings, posts_count)
    out_dir = project_dir()
    log.info("Writing %d posts into %s", len(posts), out_dir)

    for post in posts:
        post_id = post.get("id", "unknown")
        save_path = out_dir / f"post_{post_id}.txt"

        # FRESH screenshot + FRESH grounding for every single iteration.
        # Coordinates from previous runs are intentionally discarded.
        show_desktop(executor)
        result = seeker.locate(instruction)

        if result.found:
            executor.double_click(result.screen_x, result.screen_y)
        elif allow_fallback:
            log.warning("Icon not found for post %s after all retries; using OS fallback.",
                        post_id)
            if not launch_notepad_via_os():
                log.error("Both grounding and OS fallback failed; aborting.")
                return 1
        else:
            log.error("Could not locate %r for post %s; aborting (fallback disabled).",
                      instruction, post_id)
            return 1

        if not verify_notepad_launched(settings, capturer, verifier):
            log.error("Notepad did not launch for post %s; aborting.", post_id)
            return 1

        write_post_in_notepad(executor, post, save_path)
        log.info("Saved %s", save_path)

    log.info("Task complete: %d posts written to %s", len(posts), out_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ScreenSeekeR visual grounding system")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                        help="natural-language description of the target element")
    parser.add_argument("--posts", type=int, default=10, help="number of posts to write")
    parser.add_argument("--locate-only", action="store_true",
                        help="only locate + annotate the target; do not click")
    parser.add_argument("--no-fallback", action="store_true",
                        help="disable the OS-launch fallback when grounding fails")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        settings = Settings.load()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    settings.ensure_dirs()
    setup_logging(settings.debug_dir, logging.DEBUG if args.verbose else logging.INFO)

    try:
        return run_task(
            settings, args.instruction, args.posts, args.locate_only,
            allow_fallback=not args.no_fallback,
        )
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 130
    except Exception:
        log.exception("Fatal error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
