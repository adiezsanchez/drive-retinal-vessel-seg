#!/usr/bin/env python3
"""Interactive Napari viewer: fundus + ground truth + each method as layers.

On a desktop with a display this opens Napari. In headless environments it
writes the same overlay PNGs that `pixi run evaluate` produces and exits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drive_seg.data import iter_samples, load_sample, resolve_dataset_root
from drive_seg.io_pred import load_prediction
from drive_seg.paths import FIGURES_DIR, ensure_output_dirs
from drive_seg.viz import overlay_rgb, save_png

METHODS = ("frangi", "rf", "svm", "unet")
COLORMAPS = {
    "ground_truth": "green",
    "frangi": "cyan",
    "rf": "yellow",
    "svm": "magenta",
    "unet": "red",
}


def _has_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default=None, help="Defaults to the first test image.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    root = resolve_dataset_root()
    samples = iter_samples(root, split=args.split)
    sample = samples[0]
    if args.sample_id:
        matching = [s for s in samples if s.sample_id == args.sample_id]
        if not matching:
            raise SystemExit(f"Unknown sample_id {args.sample_id}")
        sample = matching[0]

    image, vessel, fov = load_sample(sample)
    preds = {}
    for method in METHODS:
        try:
            preds[method] = load_prediction(method, sample.sample_id, sample.split)
        except FileNotFoundError:
            continue

    # Always dump a static overlay so CI / headless machines leave an artifact.
    combined = overlay_rgb(image, preds.get("unet", vessel), vessel, fov)
    save_png(FIGURES_DIR / f"napari_overlay_{sample.sample_id}.png", combined)

    if args.headless or not _has_display():
        print(
            f"No GUI display (or --headless). Wrote {FIGURES_DIR / f'napari_overlay_{sample.sample_id}.png'}\n"
            "On a workstation with NVIDIA GPU + desktop, run: pixi run napari"
        )
        return

    import napari

    viewer = napari.Viewer(title=f"DRIVE vessels — {sample.sample_id}")
    viewer.add_image(image, name="fundus", rgb=True)
    viewer.add_labels(fov.astype(int), name="FOV", opacity=0.15)
    viewer.add_labels(vessel.astype(int), name="ground_truth", opacity=0.4)
    for method, pred in preds.items():
        viewer.add_labels(
            pred.astype(int),
            name=method,
            opacity=0.4,
        )
    print("Napari is open. Toggle layers in the left dock. Close the window to exit.")
    napari.run()


if __name__ == "__main__":
    main()
