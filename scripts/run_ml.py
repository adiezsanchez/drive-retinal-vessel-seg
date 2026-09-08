#!/usr/bin/env python3
"""Train pixel-wise Random Forest and linear SVM, then segment the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drive_seg.data import iter_samples, load_sample, resolve_dataset_root
from drive_seg.io_pred import save_prediction
from drive_seg.ml import (
    build_random_forest,
    build_svm,
    pixels_from_sample,
    predict_mask,
    save_model,
)
from drive_seg.paths import MODELS_DIR


def _fit(name: str, builder, train_xy: tuple[np.ndarray, np.ndarray], test_samples) -> None:
    x, y = train_xy
    print(f"Training {name} on {len(y)} balanced pixels…")
    model = builder()
    model.fit(x, y)
    save_model(model, MODELS_DIR / f"{name}.joblib")
    for sample in tqdm(test_samples, desc=f"{name}:test"):
        image, _vessel, fov = load_sample(sample)
        pred = predict_mask(model, image, fov)
        save_prediction(name, sample.sample_id, sample.split, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-class", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = resolve_dataset_root()
    rng = np.random.default_rng(args.seed)
    train_samples = iter_samples(root, split="training")
    test_samples = iter_samples(root, split="test")

    xs, ys = [], []
    for sample in tqdm(train_samples, desc="features"):
        image, vessel, fov = load_sample(sample)
        x, y = pixels_from_sample(image, vessel, fov, rng, max_per_class=args.max_per_class)
        xs.append(x)
        ys.append(y)
    train_xy = (np.concatenate(xs), np.concatenate(ys))
    print(f"Feature matrix {train_xy[0].shape}")

    _fit("rf", build_random_forest, train_xy, test_samples)
    _fit("svm", build_svm, train_xy, test_samples)
    print("Wrote RF/SVM predictions and models.")


if __name__ == "__main__":
    main()
