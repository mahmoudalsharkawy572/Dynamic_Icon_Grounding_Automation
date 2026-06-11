"""ScreenSeekeR core: recursive plan -> crop -> ground -> verify search.

Pipeline (the assignment's required architecture):

  Screenshot -> Planner -> Candidate Regions -> Recursive Search
            -> Grounding -> Verification -> Coordinate Projection -> Click
"""

from __future__ import annotations

import json
import time

import numpy as np

from src.api_client import ApiError
from src.config import Settings
from src.grounder import Grounder
from src.image_utils import (
    classify_screen_position,
    crop,
    draw_final_result,
    draw_grounding_annotation,
    extract_patch,
    project_chain,
    project_to_parent,
    save_image,
)
from src.logger import get_logger
from src.models import (
    GroundingPoint,
    PixelBox,
    Region,
    SearchAttempt,
    SearchResult,
)
from src.planner import Planner, PlannerError
from src.screenshot import ScreenCapturer
from src.verifier import Verifier

log = get_logger("screenseeker")


class ScreenSeekeR:
    """Recursive visual search orchestrator."""

    def __init__(
        self,
        settings: Settings,
        capturer: ScreenCapturer,
        planner: Planner,
        grounder: Grounder,
        verifier: Verifier,
    ) -> None:
        self._s = settings
        self._capturer = capturer
        self._planner = planner
        self._grounder = grounder
        self._verifier = verifier
        self._attempt_counters: dict[int, int] = {}
        self._trace: list[dict] = []
        self._full_screenshot: np.ndarray | None = None

    # ------------------------------------------------------------------ API

    def locate(self, instruction: str) -> SearchResult:
        """Run a complete fresh search for `instruction`.

        Takes a NEW screenshot every call — coordinates are never reused.
        Outer retry loop covers full-search failures (icon not found,
        API outages that exhausted their own retries).
        """
        last_result = SearchResult(found=False)
        for run in range(1, self._s.max_retries + 1):
            log.info("=== ScreenSeekeR search run %d/%d: %r ===",
                     run, self._s.max_retries, instruction)
            self._attempt_counters = {}
            self._trace = []
            self._full_screenshot = self._capturer.capture_primary()
            save_image(self._s.debug_dir / "full_screenshot.png", self._full_screenshot)
            try:
                result = self._search(
                    instruction=instruction,
                    image=self._full_screenshot,
                    origins=[],
                    depth=0,
                    context="full screen, depth 0",
                )
            except (PlannerError, ApiError) as exc:
                log.warning("Search run failed: %s", exc)
                result = SearchResult(found=False)
            self._dump_trace()
            if result.found:
                self._save_final_artifacts(result)
                return result
            last_result = result
            log.warning("Search run %d failed (icon not found); retrying in %.1fs",
                        run, self._s.retry_delay_s)
            time.sleep(self._s.retry_delay_s)
        return last_result

    # ------------------------------------------------------- recursive core

    def _search(
        self,
        instruction: str,
        image: np.ndarray,
        origins: list[tuple[int, int]],
        depth: int,
        context: str,
    ) -> SearchResult:
        if depth >= self._s.max_depth:
            log.info("Max depth %d reached; backtracking", self._s.max_depth)
            return SearchResult(found=False)

        h, w = image.shape[:2]
        try:
            regions = self._planner.plan(instruction, image, context=context)
        except (PlannerError, ApiError) as exc:
            # API outage or empty plan at this level: fail this branch only,
            # so the caller can backtrack to sibling regions or retry the run.
            log.warning("Planner unavailable at depth %d: %s", depth, exc)
            return SearchResult(found=False)

        Planner.save_plan(
            self._s.debug_dir / f"plan_depth{depth}.json", instruction, context, regions
        )

        deferred: list[tuple[Region, np.ndarray, list[tuple[int, int]]]] = []

        # Phase 1: ground + verify each candidate region in priority order.
        for region in regions:
            attempt_no = self._next_attempt(depth)
            box = region.to_pixel_box(w, h).expanded_to_min(self._s.min_crop_px, w, h)
            crop_res = crop(image, box)
            crop_path = self._s.debug_dir / f"crop_depth{depth}_attempt{attempt_no}.png"
            save_image(crop_path, crop_res.image)
            log.info("[depth %d attempt %d] region=%r box=(%d,%d,%d,%d) neighbors=%s",
                     depth, attempt_no, region.description,
                     box.left, box.top, box.right, box.bottom, list(region.neighbors))

            point = self._safe_ground(instruction, crop_res.image)
            grounded = point is not None
            verified = False
            point_global: tuple[int, int] | None = None

            if point is not None:
                annotated = draw_grounding_annotation(
                    crop_res.image, point, label=f"d{depth}a{attempt_no}"
                )
                save_image(
                    self._s.debug_dir / f"crop_depth{depth}_attempt{attempt_no}_annotated.png",
                    annotated,
                )
                verified, point_global = self._verify_point(
                    instruction, image, crop_res, point, origins, depth, attempt_no
                )

            self._record(depth, attempt_no, region, crop_res, grounded, verified, point_global)

            if verified and point_global is not None:
                return SearchResult(
                    found=True,
                    screen_x=point_global[0],
                    screen_y=point_global[1],
                    depth=depth,
                    region_description=region.description,
                )

            # Keep region for the recursive phase if the crop is still divisible.
            if min(crop_res.image.shape[0], crop_res.image.shape[1]) >= 2 * self._s.min_crop_px:
                deferred.append(
                    (region, crop_res.image, origins + [(crop_res.origin_x, crop_res.origin_y)])
                )

        # Phase 2: all candidates failed -> recurse deeper into them, best first.
        for region, sub_image, sub_origins in deferred:
            log.info("Recursing into region %r at depth %d", region.description, depth + 1)
            sub = self._search(
                instruction=instruction,
                image=sub_image,
                origins=sub_origins,
                depth=depth + 1,
                context=(
                    f"depth {depth + 1}; searching inside parent region "
                    f"'{region.description}' (neighbors: {', '.join(region.neighbors) or 'none'})"
                ),
            )
            if sub.found:
                return sub

        return SearchResult(found=False)

    # ------------------------------------------------------------- helpers

    def _safe_ground(self, instruction: str, crop_bgr: np.ndarray) -> GroundingPoint | None:
        try:
            return self._grounder.ground(instruction, crop_bgr)
        except ApiError as exc:  # API exhausted retries — try next region
            log.warning("Grounder failed for this crop: %s", exc)
            return None
        except Exception as exc:
            log.warning("Unexpected grounder error: %s", exc)
            return None

    def _verify_point(
        self,
        instruction: str,
        parent_image: np.ndarray,
        crop_res,
        point: GroundingPoint,
        origins: list[tuple[int, int]],
        depth: int,
        attempt_no: int,
    ) -> tuple[bool, tuple[int, int] | None]:
        """Verify on a context patch from the PARENT image around the point."""
        parent_pt = project_to_parent(point, crop_res)
        patch = extract_patch(parent_image, parent_pt, self._s.verification_patch_px)
        local = GroundingPoint(
            x=parent_pt[0] - patch.origin_x, y=parent_pt[1] - patch.origin_y
        )
        annotated_patch = draw_grounding_annotation(patch.image, local, label="candidate")
        save_image(
            self._s.debug_dir / f"verification_depth{depth}_attempt{attempt_no}.png",
            annotated_patch,
        )
        result = self._verifier.verify(instruction, annotated_patch)
        if not result.correct:
            return False, None
        screen_pt = project_chain(parent_pt, list(reversed(origins)))
        log.info("Verified target at screen point (%d,%d)", screen_pt[0], screen_pt[1])
        return True, screen_pt

    def _next_attempt(self, depth: int) -> int:
        self._attempt_counters[depth] = self._attempt_counters.get(depth, 0) + 1
        return self._attempt_counters[depth]

    def _record(self, depth, attempt_no, region, crop_res, grounded, verified, point_global) -> None:
        self._trace.append(
            SearchAttempt(
                depth=depth,
                attempt=attempt_no,
                region=region,
                crop_origin=(crop_res.origin_x, crop_res.origin_y),
                grounded=grounded,
                verified=verified,
                point_global=point_global,
            ).__dict__
            | {"region": region.to_log_dict()}
        )

    def _dump_trace(self) -> None:
        path = self._s.debug_dir / "search_trace.json"
        path.write_text(json.dumps(self._trace, indent=2, default=str), encoding="utf-8")

    def _save_final_artifacts(self, result: SearchResult) -> None:
        """final_result.png in debug/ + zone-named annotated screenshot in output/."""
        if self._full_screenshot is None:
            return
        h, w = self._full_screenshot.shape[:2]
        half = self._s.verification_patch_px // 2
        region_box = PixelBox(
            result.screen_x - half, result.screen_y - half,
            result.screen_x + half, result.screen_y + half,
        ).clamped(w, h)
        final = draw_final_result(
            self._full_screenshot, region_box, (result.screen_x, result.screen_y)
        )
        save_image(self._s.debug_dir / "final_result.png", final)
        zone = classify_screen_position(result.screen_x, result.screen_y, w, h)
        save_image(self._s.output_dir / f"{zone}_detection.png", final)
        log.info("Saved final artifacts (zone=%s)", zone)
