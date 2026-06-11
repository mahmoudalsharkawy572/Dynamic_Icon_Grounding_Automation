"""Resilient Gemini API client: retries, timeouts, and tolerant JSON extraction.

All three ScreenSeekeR roles (planner, grounder, verifier) call Gemini through
this single client so retry/parse behavior is uniform.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import cv2
import numpy as np
from google import genai
from google.genai import types

from src.config import Settings
from src.logger import get_logger

log = get_logger("api_client")


class ApiError(RuntimeError):
    """Raised when the API fails after all retries (network down, timeout, 5xx...)."""


class JsonParseError(ValueError):
    """Raised when no JSON can be recovered from a model response."""


def extract_json(text: str) -> Any:
    """Best-effort JSON recovery from an LLM response.

    Handles: clean JSON, ```json fenced blocks, and JSON embedded in prose.
    """
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Scan for the first balanced object/array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise JsonParseError(f"No parseable JSON in model response: {text[:300]!r}")


def encode_image_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to PNG-encode image")
    return buf.tobytes()


class GeminiClient:
    """Thin wrapper over google-genai with retry + JSON contract enforcement."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.request_timeout_s * 1000)
            ),
        )

    def generate_json(
        self,
        model: str,
        prompt: str,
        image_bgr: np.ndarray | None = None,
        temperature: float = 0.0,
    ) -> Any:
        """Send prompt (+ optional image), return parsed JSON.

        Retries MAX_RETRIES times with a fixed delay on transport errors,
        timeouts, and malformed-JSON responses.
        """
        parts: list[types.Part] = []
        if image_bgr is not None:
            parts.append(
                types.Part.from_bytes(
                    data=encode_image_png(image_bgr), mime_type="image/png"
                )
            )
        parts.append(types.Part.from_text(text=prompt))

        last_error: Exception | None = None
        for attempt in range(1, self._settings.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                text = response.text or ""
                return extract_json(text)
            except JsonParseError as exc:
                last_error = exc
                log.warning("Malformed JSON from %s (attempt %d/%d): %s",
                            model, attempt, self._settings.max_retries, exc)
            except Exception as exc:  # network down, timeout, quota, 5xx
                last_error = exc
                log.warning("API call to %s failed (attempt %d/%d): %s",
                            model, attempt, self._settings.max_retries, exc)
            if attempt < self._settings.max_retries:
                time.sleep(self._settings.retry_delay_s)

        raise ApiError(
            f"Gemini call to {model} failed after "
            f"{self._settings.max_retries} attempts: {last_error}"
        )
