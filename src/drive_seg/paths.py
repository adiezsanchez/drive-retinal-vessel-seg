"""Repository layout and output locations."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
DRIVE_DIR = DATA_DIR / "drive"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def ensure_output_dirs() -> None:
    for path in (FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR):
        path.mkdir(parents=True, exist_ok=True)
