"""Pixel metrics inside the field of view: Dice, IoU, sensitivity, specificity, precision."""

from __future__ import annotations

from typing import Any

import numpy as np


def _counts(pred: np.ndarray, truth: np.ndarray, fov: np.ndarray) -> tuple[int, int, int, int]:
    p = np.asarray(pred, dtype=bool) & np.asarray(fov, dtype=bool)
    t = np.asarray(truth, dtype=bool) & np.asarray(fov, dtype=bool)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t & np.asarray(fov, dtype=bool)))
    return tp, fp, fn, tn


def _div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def segmentation_metrics(pred: np.ndarray, truth: np.ndarray, fov: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = _counts(pred, truth, fov)
    return {
        "dice": _div(2 * tp, 2 * tp + fp + fn),
        "iou": _div(tp, tp + fp + fn),
        "sensitivity": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "precision": _div(tp, tp + fp),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def mean_metrics(rows: list[dict[str, Any]], keys: tuple[str, ...] | None = None) -> dict[str, float]:
    keys = keys or ("dice", "iou", "sensitivity", "specificity", "precision")
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows]
        out[key] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return out
