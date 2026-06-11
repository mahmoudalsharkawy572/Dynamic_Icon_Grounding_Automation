"""Grounder role (Gemini Robotics ER 1.6 Preview): pixel-precise pointing.

Receives ONLY a cropped candidate region (never the full screen), which is the
core ScreenSeekeR idea: grounding accuracy rises sharply on small crops.
"""

from __future__ import annotations

import numpy as np

from src.api_client import GeminiClient
from src.config import Settings
from src.image_utils import normalized_to_pixel
from src.logger import get_logger
from src.models import GroundingPoint

log = get_logger("grounder")

_PROMPT = """You are a precise GUI grounding model.
Point to the following target in the image: {instruction}

The image is a cropped region of a desktop screen. If the target (or its text
label) is visible, return the point at its visual center.

Respond with JSON ONLY, coordinates normalized to a 0-1000 scale:
{{"x": <int 0-1000>, "y": <int 0-1000>}}

If the target is definitely NOT visible in this crop, respond with:
{{"x": -1, "y": -1}}
"""


class Grounder:
    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def ground(self, instruction: str, crop_bgr: np.ndarray) -> GroundingPoint | None:
        """Return the grounding point in crop-pixel coordinates, or None."""
        h, w = crop_bgr.shape[:2]
        data = self._client.generate_json(
            self._settings.grounder_model,
            _PROMPT.format(instruction=instruction),
            crop_bgr,
        )
        norm = self._parse_point(data)
        if norm is None:
            log.info("Grounder reports target not visible in crop (%dx%d)", w, h)
            return None
        px, py = normalized_to_pixel(norm[0], norm[1], w, h)
        log.info("Grounder point: norm=(%.0f,%.0f) -> crop px=(%d,%d)",
                 norm[0], norm[1], px, py)
        return GroundingPoint(x=px, y=py)

    @staticmethod
    def _parse_point(data: object) -> tuple[float, float] | None:
        """Safely parse {"x":..,"y":..} or Robotics-style [{"point":[y,x]}]."""
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return None
        if "point" in data and isinstance(data["point"], (list, tuple)) \
                and len(data["point"]) >= 2:
            y, x = data["point"][0], data["point"][1]  # robotics convention [y, x]
            try:
                x_f, y_f = float(x), float(y)
            except (TypeError, ValueError):
                return None
        elif "x" in data and "y" in data:
            try:
                x_f, y_f = float(data["x"]), float(data["y"])
            except (TypeError, ValueError):
                return None
        else:
            return None
        if x_f < 0 or y_f < 0:
            return None  # explicit "not visible" sentinel
        return x_f, y_f
