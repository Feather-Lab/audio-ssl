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

FMRI_COMPONENTS_SUMMARY_CSV = (
    "PLOTS_across-models/"
    "across-models_barplot_components_NH2015comp_CV-splits-nit-10_"
    "median_r2_test_sem_over_it_median_r2_test_performance_sorted.csv"
)

AUDITORY_BRAIN_DNN_MARKER = Path("aud_dnn/analyze/plot_utils_AUD.py")


def discover_fmri_results_dir() -> Path | None:
    """Locate auditory_brain_dnn_for_audio_ssl ridge-regression results."""
    if "COCHDNN_FMRI_RESULTS_DIR" in os.environ:
        return Path(os.environ["COCHDNN_FMRI_RESULTS_DIR"])

    marker = Path(FMRI_COMPONENTS_SUMMARY_CSV)
    candidates = [
        WORKING_DIRECTORY.parent / "auditory_brain_dnn_for_audio_ssl" / "results",
        WORKING_DIRECTORY / "auditory_brain_dnn" / "results",
    ]
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate

    parts = WORKING_DIRECTORY.parts
    if "users" in parts:
        users_root = Path(*parts[: parts.index("users") + 1])
        for results_dir in users_root.glob(
            "*/projects/auditory_brain_dnn_for_audio_ssl/results"
        ):
            if (results_dir / marker).is_file():
                return results_dir
    return None


FMRI_RESULTS_DIR = discover_fmri_results_dir()


def discover_auditory_brain_dnn_root() -> Path | None:
    """Locate auditory_brain_dnn code checkout containing analysis helpers."""
    if "COCHDNN_AUDITORY_BRAIN_DNN_DIR" in os.environ:
        candidate = Path(os.environ["COCHDNN_AUDITORY_BRAIN_DNN_DIR"])
        if (candidate / AUDITORY_BRAIN_DNN_MARKER).is_file():
            return candidate
        if (candidate / "plot_utils_AUD.py").is_file():
            return candidate.parent.parent

    candidates = [
        WORKING_DIRECTORY.parent / "auditory_brain_dnn_for_audio_ssl",
        WORKING_DIRECTORY / "auditory_brain_dnn",
    ]
    for candidate in candidates:
        if (candidate / AUDITORY_BRAIN_DNN_MARKER).is_file():
            return candidate

    parts = WORKING_DIRECTORY.parts
    if "users" in parts:
        users_root = Path(*parts[: parts.index("users") + 1])
        for repo_root in users_root.glob(
            "*/projects/auditory_brain_dnn_for_audio_ssl"
        ):
            if (repo_root / AUDITORY_BRAIN_DNN_MARKER).is_file():
                return repo_root
    return None


AUDITORY_BRAIN_DNN_ROOT = discover_auditory_brain_dnn_root()


def require_path(path: Path | None, env_var: str, description: str) -> Path:
    """Return a configured path or raise a clear setup error."""
    if path is None:
        raise RuntimeError(
            f"{description} is not configured. Set {env_var} to the local path."
        )
    return path
