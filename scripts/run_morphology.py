#!/usr/bin/env python3
"""Vessel morphology (density, length, width, branching, tortuosity) from predicted masks."""

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
from drive_seg.morphology import morphology_metrics
from drive_seg.paths import FIGURES_DIR, METRICS_DIR, ensure_output_dirs
from drive_seg.viz import morphology_chart

METHODS = ("frangi", "rf", "svm", "unet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    ensure_output_dirs()
    root = resolve_dataset_root()
    samples = iter_samples(root, split=args.split)

    rows = []
    for sample in samples:
        _image, vessel, fov = load_sample(sample)
        truth = morphology_metrics(vessel, fov)
        truth.update({"method": "ground_truth", "sample_id": sample.sample_id})
        rows.append(truth)
        for method in METHODS:
            try:
                pred = load_prediction(method, sample.sample_id, sample.split)
            except FileNotFoundError:
                continue
            m = morphology_metrics(pred, fov)
            m.update({"method": method, "sample_id": sample.sample_id})
            rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "morphology_per_image.csv", index=False)
    agg = df.groupby("method").mean(numeric_only=True).reset_index()
    agg.to_csv(METRICS_DIR / "morphology_summary.csv", index=False)
    (METRICS_DIR / "morphology_summary.json").write_text(json.dumps(agg.to_dict("records"), indent=2))
    plot_df = df[df["method"] != "ground_truth"]
    if not plot_df.empty:
        morphology_chart(
            plot_df,
            FIGURES_DIR / "morphology_comparison.html",
            FIGURES_DIR / "morphology_comparison.png",
        )
    print(agg.to_string(index=False))
    print(f"Wrote morphology tables under {METRICS_DIR}")


if __name__ == "__main__":
    main()
