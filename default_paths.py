"""Repository and dataset paths used by release scripts.

No lab-local paths are hardcoded here. Set the ``COCHDNN_*`` environment
variables below to point scripts at datasets or checkpoints on your system.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKING_DIRECTORY = Path(__file__).resolve().parent
MODEL_CHECKPOINT_DIR = Path(
    os.environ.get("COCHDNN_CHECKPOINT_DIR", WORKING_DIRECTORY / "model_checkpoints")
)
MODEL_DIRECTORY = Path(os.environ.get("COCHDNN_MODEL_DIR", WORKING_DIRECTORY / "model_directories"))

DATA_ROOT = Path(os.environ["COCHDNN_DATA_ROOT"]) if "COCHDNN_DATA_ROOT" in os.environ else None
JSIN_PATH = Path(os.environ["COCHDNN_JSIN_DIR"]) if "COCHDNN_JSIN_DIR" in os.environ else None
ESC50_DIR = Path(os.environ["COCHDNN_ESC50_DIR"]) if "COCHDNN_ESC50_DIR" in os.environ else None
NSYNTH_DIR = Path(os.environ["COCHDNN_NSYNTH_DIR"]) if "COCHDNN_NSYNTH_DIR" in os.environ else None
TONE_PERFECT_DIR = (
    Path(os.environ["COCHDNN_TONE_PERFECT_DIR"])
    if "COCHDNN_TONE_PERFECT_DIR" in os.environ
    else None
)


def require_path(path: Path | None, env_var: str, description: str) -> Path:
    """Return a configured path or raise a clear setup error."""
    if path is None:
        raise RuntimeError(
            f"{description} is not configured. Set {env_var} to the local path."
        )
    return path
