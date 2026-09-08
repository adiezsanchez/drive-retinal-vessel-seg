"""Classical vessel segmentation: Frangi vesselness + Otsu threshold.

Frangi et al. (MICCAI 1998) measure how Hessian eigenvalues match a tubular
structure across scales. On fundus photographs we run it on the inverted,
CLAHE-enhanced green channel, restrict to the FOV, then threshold.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import binary_opening, disk, remove_small_objects

from .preprocess import prepare_gray


def vesselness(
    image: np.ndarray,
    sigmas: tuple[float, ...] = (0.8, 1.2, 1.6, 2.2, 3.0, 4.0),
) -> np.ndarray:
    gray = prepare_gray(image, invert=True)
    resp = frangi(gray, sigmas=sigmas, black_ridges=False)
    resp = resp.astype(np.float32)
    finite = np.isfinite(resp)
    if not finite.any():
        return np.zeros_like(gray, dtype=np.float32)
    resp[~finite] = 0.0
    peak = float(resp.max())
    if peak > 0:
        resp /= peak
    return resp


def segment_frangi(
    image: np.ndarray,
    fov: np.ndarray,
    min_size: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(binary_mask, vesselness_map)`` both restricted to ``fov``."""
    resp = vesselness(image)
    resp_fov = resp.copy()
    resp_fov[~fov] = 0.0
    values = resp_fov[fov]
    if values.size == 0 or float(values.max()) <= 0:
        empty = np.zeros(fov.shape, dtype=bool)
        return empty, resp_fov
    # Ignore near-zero background when choosing Otsu on a sparse vesselness map.
    positive = values[values > np.percentile(values, 70)]
    thresh = float(threshold_otsu(positive)) if positive.size > 20 else float(np.percentile(values, 95))
    binary = (resp_fov >= thresh) & fov
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=min_size)
    return binary.astype(bool), resp_fov
