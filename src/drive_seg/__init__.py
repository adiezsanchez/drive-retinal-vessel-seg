"""DRIVE retinal vessel segmentation teaching library."""

from .device import get_device, describe_device
from .paths import REPO_ROOT, FIGURES_DIR, METRICS_DIR

__all__ = ["get_device", "describe_device", "REPO_ROOT", "FIGURES_DIR", "METRICS_DIR"]
__version__ = "0.1.0"
