"""Fundus preprocessing used by every method.

Retinal vessels are most contrasted in the green channel. Contrast-limited
adaptive histogram equalization (CLAHE) is a standard DRIVE preprocessing
step before Frangi, pixel classifiers, or a U-Net.
"""

from __future__ import annotations

import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_float32


def green_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return img_as_float32(image)
    return img_as_float32(image[..., 1])


def clahe(gray: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    gray = img_as_float32(gray)
    return equalize_adapthist(gray, clip_limit=clip_limit).astype(np.float32)


def invert_vessels_bright(gray: np.ndarray) -> np.ndarray:
    """Frangi and many ridge filters expect bright tubular structures."""
    return (1.0 - img_as_float32(gray)).astype(np.float32)


def prepare_gray(image: np.ndarray, invert: bool = True) -> np.ndarray:
    gray = clahe(green_channel(image))
    if invert:
        gray = invert_vessels_bright(gray)
    return gray
