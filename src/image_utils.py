"""Cropping, coordinate projection, annotation, and artifact saving."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.models import CropResult, GroundingPoint, PixelBox

GREEN = (0, 255, 0)  # BGR
RED = (0, 0, 255)


def crop(image: np.ndarray, box: PixelBox) -> CropResult:
    """Crop a region; remember its origin for coordinate projection."""
    h, w = image.shape[:2]
    box = box.clamped(w, h)
    return CropResult(
        image=image[box.top : box.bottom, box.left : box.right].copy(),
        origin_x=box.left,
        origin_y=box.top,
    )


def project_to_parent(point: GroundingPoint, crop_result: CropResult) -> tuple[int, int]:
    """Project a crop-local point into the crop's parent coordinate space.

    Example: crop origin (400, 700) + grounded point (543, 973) -> (943, 1673).
    """
    return crop_result.origin_x + point.x, crop_result.origin_y + point.y


def project_chain(point_xy: tuple[int, int], origins: list[tuple[int, int]]) -> tuple[int, int]:
    """Project through a chain of nested crop origins back to global screen space."""
    x, y = point_xy
    for ox, oy in origins:
        x, y = x + ox, y + oy
    return x, y


def normalized_to_pixel(x_norm: float, y_norm: float, w: int, h: int,
                        scale: float = 1000.0) -> tuple[int, int]:
    """Convert [0, scale] normalized model coordinates to pixel coordinates."""
    px = int(round(min(max(x_norm, 0.0), scale) / scale * (w - 1)))
    py = int(round(min(max(y_norm, 0.0), scale) / scale * (h - 1)))
    return px, py


def classify_screen_position(x: int, y: int, screen_w: int, screen_h: int) -> str:
    """Bucket a screen point into a named zone (used for output/ filenames)."""
    col = 0 if x < screen_w / 3 else (1 if x < 2 * screen_w / 3 else 2)
    row = 0 if y < screen_h / 3 else (1 if y < 2 * screen_h / 3 else 2)
    names = {
        (0, 0): "top_left", (1, 0): "top_center", (2, 0): "top_right",
        (0, 1): "center_left", (1, 1): "center", (2, 1): "center_right",
        (0, 2): "bottom_left", (1, 2): "bottom_center", (2, 2): "bottom_right",
    }
    return names[(col, row)]


def draw_grounding_annotation(
    image: np.ndarray,
    point: GroundingPoint,
    label: str = "",
    box_half: int = 40,
) -> np.ndarray:
    """Green rectangle around the predicted point + green center crosshair + text."""
    out = image.copy()
    h, w = out.shape[:2]
    x, y = int(point.x), int(point.y)
    cv2.rectangle(
        out,
        (max(0, x - box_half), max(0, y - box_half)),
        (min(w - 1, x + box_half), min(h - 1, y + box_half)),
        GREEN, 2,
    )
    cv2.drawMarker(out, (x, y), GREEN, cv2.MARKER_CROSS, 24, 2)
    cv2.circle(out, (x, y), 4, GREEN, -1)
    text = f"{label} ({x},{y})".strip()
    cv2.putText(out, text, (max(4, x - box_half), max(20, y - box_half - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    return out


def draw_final_result(
    full_screenshot: np.ndarray,
    region_box: PixelBox,
    screen_point: tuple[int, int],
) -> np.ndarray:
    """Final artifact: full screenshot + selected region + target + click position."""
    out = full_screenshot.copy()
    cv2.rectangle(out, (region_box.left, region_box.top),
                  (region_box.right, region_box.bottom), GREEN, 3)
    cv2.putText(out, "selected region", (region_box.left + 6, max(24, region_box.top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
    x, y = screen_point
    cv2.drawMarker(out, (x, y), RED, cv2.MARKER_CROSS, 36, 3)
    cv2.circle(out, (x, y), 8, GREEN, 2)
    cv2.putText(out, f"click ({x},{y})", (x + 14, y - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)
    return out


def extract_patch(image: np.ndarray, center: tuple[int, int], size: int) -> CropResult:
    """Square context patch around a point (for the verifier)."""
    h, w = image.shape[:2]
    half = size // 2
    box = PixelBox(center[0] - half, center[1] - half,
                   center[0] + half, center[1] + half).clamped(w, h)
    return crop(image, box)


def save_image(path: Path, image: np.ndarray) -> None:
    """Save, creating parent dirs. Existing files are overwritten deliberately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
