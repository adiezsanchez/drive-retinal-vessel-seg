#!/usr/bin/env python3
"""Verify the teaching stack: PyTorch (+CUDA if present), Napari, Plotly. No TensorFlow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _ok(name: str, extra: str = "") -> None:
    print(f"  OK  {name}{extra}")


def main() -> int:
    print("Stack check (CUDA primary, CPU fallback; Napari; Plotly-only charts)")

    import torch

    _ok("torch", f" {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"       GPU: {torch.cuda.get_device_name(0)}  cuda {torch.version.cuda}")
    else:
        print("       CPU fallback — fine for the demo; CUDA is the documented primary path.")

    import napari

    _ok("napari", f" {napari.__version__}")

    import plotly

    _ok("plotly", f" {plotly.__version__}")

    import sklearn
    import skimage

    _ok("scikit-learn", f" {sklearn.__version__}")
    _ok("scikit-image", f" {skimage.__version__}")

    if importlib.util.find_spec("tensorflow") is not None:
        print("  note TensorFlow is importable on this machine but is not used by this project.")
    else:
        _ok("no TensorFlow dependency (as required)")

    from drive_seg.device import describe_device
    from drive_seg.viz import metrics_bar_chart  # noqa: F401 — plotly path, not matplotlib

    print(f"device: {describe_device()}")
    print("Charts go through drive_seg.viz (Plotly). Matplotlib/seaborn are not used for plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
