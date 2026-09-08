#!/usr/bin/env python3
"""End-to-end demo: synthetic data → Frangi → RF/SVM → U-Net → metrics → morphology → figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drive_seg.paths import FIGURES_DIR, SYNTHETIC_DIR, ensure_output_dirs
from drive_seg.synthetic import generate_synthetic_dataset


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *(extra or [])]
    print(f"\n=== {script} {' '.join(extra or [])}===".strip())
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    ensure_output_dirs()
    if not (SYNTHETIC_DIR / "training" / "images").is_dir():
        print("Generating synthetic DRIVE-layout demo data…")
        generate_synthetic_dataset()
    else:
        print(f"Using existing synthetic data at {SYNTHETIC_DIR}")

    _run("check_stack.py")
    _run("run_classical.py")
    _run("run_ml.py")
    _run("run_unet.py")
    _run("run_evaluate.py")
    _run("run_morphology.py")
    _run("visualize_napari.py", ["--headless"])

    print("\nDemo finished.")
    print(f"Figures: {FIGURES_DIR}")
    print("Open the Plotly HTML files in a browser. Interactive layers: pixi run napari")


if __name__ == "__main__":
    main()
