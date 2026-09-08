"""Persist binary predictions as PNG masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .paths import PREDICTIONS_DIR


def prediction_path(method: str, sample_id: str, split: str) -> Path:
    return PREDICTIONS_DIR / method / split / f"{sample_id}.png"


def save_prediction(method: str, sample_id: str, split: str, mask: np.ndarray) -> Path:
    path = prediction_path(method, sample_id, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


def load_prediction(method: str, sample_id: str, split: str) -> np.ndarray:
    path = prediction_path(method, sample_id, split)
    if not path.is_file():
        raise FileNotFoundError(path)
    from PIL import Image as _Image

    with _Image.open(path) as img:
        return np.asarray(img.convert("L")) > 0
