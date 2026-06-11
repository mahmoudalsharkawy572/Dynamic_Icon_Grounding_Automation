"""Planner role (Gemini 2.5 Flash): ScreenSeekeR position inference.

Given an instruction + screenshot, predicts ranked candidate regions with
neighbor anchors. This is what replaces naive whole-screen grounding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.api_client import GeminiClient
from src.config import PROMPTS_DIR, Settings
from src.logger import get_logger
from src.models import NORM_SCALE, Region

log = get_logger("planner")


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce any usable region."""


class Planner:
    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._template = (PROMPTS_DIR / "position_inference.txt").read_text(encoding="utf-8")

    def plan(
        self,
        instruction: str,
        image_bgr: np.ndarray,
        context: str = "full screen, depth 0",
    ) -> list[Region]:
        """Return candidate regions sorted by priority (best first)."""
        h, w = image_bgr.shape[:2]
        prompt = self._template.format(
            instruction=instruction,
            context=context,
            width=w,
            height=h,
            max_regions=self._settings.max_regions_per_level,
        )
        data = self._client.generate_json(
            self._settings.planner_model, prompt, image_bgr
        )
        regions = self._parse_regions(data)
        if not regions:
            raise PlannerError(f"Planner returned no valid regions: {data!r}")

        regions.sort(key=lambda r: r.priority)
        regions = regions[: self._settings.max_regions_per_level]
        log.info(
            "Planner output (%s): %s",
            context,
            json.dumps([r.to_log_dict() for r in regions]),
        )
        return regions

    @staticmethod
    def _parse_regions(data: object) -> list[Region]:
        """Tolerant parsing of the planner JSON contract."""
        if isinstance(data, list):
            raw_regions = data
        elif isinstance(data, dict):
            raw_regions = data.get("regions", [])
        else:
            return []

        regions: list[Region] = []
        for idx, item in enumerate(raw_regions):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox") or item.get("bbox_norm") or item.get("box")
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or not all(isinstance(v, (int, float)) for v in bbox)
            ):
                continue
            x0, y0, x1, y1 = (float(min(max(v, 0), NORM_SCALE)) for v in bbox)
            if x1 <= x0 or y1 <= y0:
                continue
            neighbors = item.get("neighbors", [])
            if not isinstance(neighbors, list):
                neighbors = []
            regions.append(
                Region(
                    description=str(item.get("description", f"region_{idx}")),
                    priority=int(item.get("priority", idx + 1)),
                    neighbors=tuple(str(n) for n in neighbors),
                    bbox_norm=(x0, y0, x1, y1),
                )
            )
        return regions

    @staticmethod
    def save_plan(path: Path, instruction: str, context: str, regions: list[Region]) -> None:
        """Persist planner output as a debug artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "instruction": instruction,
            "context": context,
            "regions": [r.to_log_dict() for r in regions],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
