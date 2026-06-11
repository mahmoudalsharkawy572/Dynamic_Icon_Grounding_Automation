"""Verifier role (Gemini 2.5 Flash): ScreenSeekeR result checking.

Shown an annotated context patch (green marker on the candidate point) and
asked whether the marker is on the correct element. Rejections trigger the
next candidate region or deeper recursion.
"""

from __future__ import annotations

import numpy as np

from src.api_client import GeminiClient
from src.config import PROMPTS_DIR, Settings
from src.logger import get_logger
from src.models import VerificationResult

log = get_logger("verifier")


class Verifier:
    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._template = (PROMPTS_DIR / "verification.txt").read_text(encoding="utf-8")

    def verify(self, instruction: str, annotated_bgr: np.ndarray) -> VerificationResult:
        prompt = self._template.format(instruction=instruction)
        try:
            data = self._client.generate_json(
                self._settings.verifier_model, prompt, annotated_bgr
            )
        except Exception as exc:  # API failure => treat as not verified, keep searching
            log.warning("Verification call failed, treating as rejection: %s", exc)
            return VerificationResult(correct=False, reason=f"verifier error: {exc}")

        if isinstance(data, dict):
            correct = bool(data.get("correct", False))
            reason = str(data.get("reason", ""))
        else:
            correct, reason = False, f"unexpected verifier payload: {data!r}"
        log.info("Verification: correct=%s reason=%s", correct, reason)
        return VerificationResult(correct=correct, reason=reason)
