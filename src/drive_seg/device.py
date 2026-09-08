"""Device selection: NVIDIA CUDA is primary, CPU is the documented fallback."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return CUDA when a GPU is visible, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return (
            f"cuda (primary) — {name}; "
            f"torch {torch.__version__}; "
            f"cuda runtime {torch.version.cuda}"
        )
    return (
        f"cpu (fallback; no NVIDIA GPU visible) — torch {torch.__version__}. "
        "Install NVIDIA drivers and `pixi install` on linux-64-cuda / win-64-cuda "
        "to use CUDA 12.x."
    )
