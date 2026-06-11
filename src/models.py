"""Typed domain models shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

NORM_SCALE = 1000  # Gemini-style normalized coordinate space [0, 1000]


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box in pixel coordinates of some image."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def clamped(self, img_w: int, img_h: int) -> "PixelBox":
        return PixelBox(
            left=max(0, min(self.left, img_w - 1)),
            top=max(0, min(self.top, img_h - 1)),
            right=max(1, min(self.right, img_w)),
            bottom=max(1, min(self.bottom, img_h)),
        )

    def expanded_to_min(self, min_px: int, img_w: int, img_h: int) -> "PixelBox":
        """Grow the box symmetrically so both sides are at least min_px."""
        grow_w = max(0, min_px - self.width)
        grow_h = max(0, min_px - self.height)
        return PixelBox(
            left=self.left - grow_w // 2,
            top=self.top - grow_h // 2,
            right=self.right + (grow_w - grow_w // 2),
            bottom=self.bottom + (grow_h - grow_h // 2),
        ).clamped(img_w, img_h)


@dataclass(frozen=True)
class Region:
    """A planner-predicted candidate region (ScreenSeekeR position inference output)."""

    description: str
    priority: int
    neighbors: tuple[str, ...]
    bbox_norm: tuple[float, float, float, float]  # x0,y0,x1,y1 in [0,1000]

    def to_pixel_box(self, img_w: int, img_h: int) -> PixelBox:
        x0, y0, x1, y1 = self.bbox_norm
        return PixelBox(
            left=int(round(x0 / NORM_SCALE * img_w)),
            top=int(round(y0 / NORM_SCALE * img_h)),
            right=int(round(x1 / NORM_SCALE * img_w)),
            bottom=int(round(y1 / NORM_SCALE * img_h)),
        ).clamped(img_w, img_h)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "priority": self.priority,
            "neighbors": list(self.neighbors),
            "bbox_norm": list(self.bbox_norm),
        }


@dataclass(frozen=True)
class CropResult:
    """A cropped sub-image plus its origin within the parent image."""

    image: np.ndarray
    origin_x: int
    origin_y: int


@dataclass(frozen=True)
class GroundingPoint:
    """Point predicted by the grounder, in pixel coords of the crop it was given."""

    x: int
    y: int


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    reason: str = ""


@dataclass(frozen=True)
class SearchAttempt:
    """One grounding attempt — used for the debug trace."""

    depth: int
    attempt: int
    region: Region
    crop_origin: tuple[int, int]
    grounded: bool
    verified: bool
    point_global: tuple[int, int] | None = None


@dataclass
class SearchResult:
    """Final outcome of a ScreenSeekeR search."""

    found: bool
    screen_x: int = -1
    screen_y: int = -1
    depth: int = -1
    region_description: str = ""
    attempts: list[SearchAttempt] = field(default_factory=list)
