"""Screen capture via MSS, returned as OpenCV-native BGR numpy arrays."""

from __future__ import annotations

import numpy as np
import mss

from src.logger import get_logger

log = get_logger("screenshot")


class ScreenCapturer:
    """Captures the primary monitor. A fresh capture is taken for every search."""

    def capture_primary(self) -> np.ndarray:
        """Return a BGR image of the primary monitor (no caching, ever)."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # index 0 = virtual all-monitors bbox
            raw = sct.grab(monitor)
            frame = np.asarray(raw, dtype=np.uint8)  # BGRA
            bgr = frame[:, :, :3].copy()
        log.debug("Captured screenshot %dx%d", bgr.shape[1], bgr.shape[0])
        return bgr
