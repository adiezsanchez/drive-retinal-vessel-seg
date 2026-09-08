#!/usr/bin/env python3
"""Frangi vesselness + Otsu threshold on the active dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from drive_seg.classical import segment_frangi
from drive_seg.data import iter_samples, load_sample, resolve_dataset_root
from drive_seg.io_pred import save_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["training", "test", "both"])
    args = parser.parse_args()
    root = resolve_dataset_root()
    splits = ["training", "test"] if args.split == "both" else [args.split]
    print(f"Classical Frangi on {root} splits={splits}")
    for split in splits:
        samples = iter_samples(root, split=split)
        for sample in tqdm(samples, desc=f"frangi:{split}"):
            image, _vessel, fov = load_sample(sample)
            pred, _resp = segment_frangi(image, fov)
            save_prediction("frangi", sample.sample_id, split, pred)
    print("Wrote predictions to results/predictions/frangi/")


if __name__ == "__main__":
    main()
