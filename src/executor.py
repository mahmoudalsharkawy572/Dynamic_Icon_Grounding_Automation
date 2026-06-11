"""Executor role: PyAutoGUI actions with smooth movement."""

from __future__ import annotations

import time

import pyautogui
import pyperclip

from src.config import Settings
from src.logger import get_logger

log = get_logger("executor")

pyautogui.FAILSAFE = True   # slam cursor into a corner to abort
pyautogui.PAUSE = 0.15


class Executor:
    def __init__(self, settings: Settings) -> None:
        self._duration = settings.mouse_move_duration_s

    def move_to(self, x: int, y: int) -> None:
        log.info("move_to (%d,%d)", x, y)
        pyautogui.moveTo(x, y, duration=self._duration, tween=pyautogui.easeInOutQuad)

    def click(self, x: int, y: int) -> None:
        log.info("click (%d,%d)", x, y)
        self.move_to(x, y)
        pyautogui.click()

    def double_click(self, x: int, y: int) -> None:
        log.info("double_click (%d,%d)", x, y)
        self.move_to(x, y)
        pyautogui.doubleClick()

    def right_click(self, x: int, y: int) -> None:
        log.info("right_click (%d,%d)", x, y)
        self.move_to(x, y)
        pyautogui.rightClick()

    def hotkey(self, *keys: str) -> None:
        log.info("hotkey %s", "+".join(keys))
        pyautogui.hotkey(*keys)

    def press(self, key: str) -> None:
        pyautogui.press(key)

    def paste_text(self, text: str) -> None:
        """Insert text via clipboard — fast and unicode/newline safe."""
        pyperclip.copy(text)
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "v")

    def type_text(self, text: str, interval: float = 0.02) -> None:
        pyautogui.typewrite(text, interval=interval)
