"""Central configuration. All tunables live here; secrets come from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when mandatory configuration is missing."""


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
DEBUG_DIR: Path = PROJECT_ROOT / "debug"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the whole pipeline."""

    gemini_api_key: str

    # Models (ScreenSeekeR roles)
    planner_model: str = "gemini-2.5-flash"
    grounder_model: str = "gemini-robotics-er-1.6-preview"
    verifier_model: str = "gemini-2.5-flash"

    # Retry policy (applies to planner, grounder, verifier, network)
    max_retries: int = 3
    retry_delay_s: float = 1.0
    request_timeout_s: float = 60.0

    # Recursive search policy
    max_depth: int = 2
    max_regions_per_level: int = 4
    min_crop_px: int = 96          # stop recursing below this crop size
    verification_patch_px: int = 320  # context window shown to the verifier

    # Executor
    mouse_move_duration_s: float = 0.4

    # Artifact directories
    debug_dir: Path = field(default=DEBUG_DIR)
    output_dir: Path = field(default=OUTPUT_DIR)

    @staticmethod
    def load() -> "Settings":
        """Build settings from environment variables.

        Required:  GEMINI_API_KEY
        Optional:  PLANNER_MODEL, GROUNDER_MODEL, VERIFIER_MODEL
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "GEMINI_API_KEY environment variable is not set. "
                "Export it before running, e.g. `export GEMINI_API_KEY=...`."
            )
        return Settings(
            gemini_api_key=api_key,
            planner_model=os.environ.get("PLANNER_MODEL", Settings.planner_model),
            grounder_model=os.environ.get("GROUNDER_MODEL", Settings.grounder_model),
            verifier_model=os.environ.get("VERIFIER_MODEL", Settings.verifier_model),
        )

    def ensure_dirs(self) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
