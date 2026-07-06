from __future__ import annotations
"""Shared project paths, device selection, and reproducibility helpers."""

import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CCPD_DIR = DATA_DIR / "ccpd"
CCPD_RAW_DIR = CCPD_DIR / "raw"
CCPD_LABELS_PATH = CCPD_DIR / "labels.csv"
SELF_PHOTOS_DIR = DATA_DIR / "self_photos"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_PATH = MODELS_DIR / "license_plate_crnn_best.pth"


def resolve_project_path(path: Path | str) -> Path:
    """Resolve relative paths against the project root."""

    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dirs() -> None:
    """Create the standard data/model/result directories used by scripts."""

    for path in [
        CCPD_RAW_DIR,
        CCPD_DIR / "train",
        CCPD_DIR / "val",
        CCPD_DIR / "test",
        SELF_PHOTOS_DIR / "raw",
        SELF_PHOTOS_DIR / "cropped",
        MODELS_DIR,
        RESULTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_device(requested: str = "auto"):
    """Return the requested torch device, or choose MPS/CUDA/CPU automatically."""

    import torch

    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for more repeatable experiments."""

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def missing_labels_message(labels_path: Path) -> str:
    """Build a helpful error message for missing ``labels.csv``."""

    try:
        rel = labels_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = labels_path
    return (
        f"Labels file not found: {labels_path}\n"
        "Please prepare the real CCPD dataset first:\n"
        f"python src/prepare_ccpd.py --raw-dir data/ccpd/raw --output {rel}"
    )
