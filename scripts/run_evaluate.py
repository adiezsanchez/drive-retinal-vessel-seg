#!/usr/bin/env python3
"""Score saved predictions (Dice / IoU / sensitivity / specificity / precision)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drive_seg.data import iter_samples, load_sample, resolve_dataset_root
from drive_seg.io_pred import load_prediction
from drive_seg.metrics import mean_metrics, segmentation_metrics
from drive_seg.paths import FIGURES_DIR, METRICS_DIR, ensure_output_dirs
from drive_seg.viz import metrics_bar_chart, overlay_figure, overlay_rgb, save_png

METHODS = ("frangi", "rf", "svm", "unet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    ensure_output_dirs()
    root = resolve_dataset_root()
    samples = iter_samples(root, split=args.split)

    rows = []
    overlay_done = False
    for sample in samples:
        image, vessel, fov = load_sample(sample)
        preds = {}
        for method in METHODS:
            try:
                pred = load_prediction(method, sample.sample_id, sample.split)
            except FileNotFoundError:
                continue
            preds[method] = pred
            m = segmentation_metrics(pred, vessel, fov)
            m.update({"method": method, "sample_id": sample.sample_id, "split": sample.split})
            rows.append(m)
        if preds and not overlay_done:
            overlay_figure(
                image,
                vessel,
                preds,
                fov,
                FIGURES_DIR / "overlays_comparison.html",
            )
            for method, pred in preds.items():
                save_png(
                    FIGURES_DIR / f"overlay_{method}_{sample.sample_id}.png",
                    overlay_rgb(image, pred, vessel, fov),
                )
            save_png(FIGURES_DIR / f"fundus_{sample.sample_id}.png", image)
            overlay_done = True

    if not rows:
        raise SystemExit("No predictions found. Run classical / ml / unet first (or `pixi run demo`).")

    per_image = pd.DataFrame(rows)
    per_image.to_csv(METRICS_DIR / "per_image.csv", index=False)
    summary_rows = []
    for method, grp in per_image.groupby("method"):
        stats = mean_metrics(grp.to_dict("records"))
        stats["method"] = method
        stats["n"] = len(grp)
        summary_rows.append(stats)
    summary = pd.DataFrame(summary_rows).sort_values("method")
    summary.to_csv(METRICS_DIR / "summary.csv", index=False)
    (METRICS_DIR / "summary.json").write_text(json.dumps(summary.to_dict("records"), indent=2))
    metrics_bar_chart(
        summary,
        FIGURES_DIR / "metrics_comparison.html",
        FIGURES_DIR / "metrics_comparison.png",
    )
    print(summary[["method", "dice", "iou", "sensitivity", "specificity", "precision"]].to_string(index=False))
    print(f"Wrote {METRICS_DIR / 'summary.csv'} and Plotly charts under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
