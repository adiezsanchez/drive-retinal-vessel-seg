"""Morphology on a binary vessel map: density, length, width, branching, tortuosity.

These are the quantities a clinician actually cares about after a mask is
produced. Thin false positives wreck length and branching counts even when Dice
looks acceptable — a useful teaching point.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant")


def skeleton_length_px(skel: np.ndarray) -> float:
    """Calibrated 8-connected length: orthogonal steps count 1, diagonals √2."""
    if not skel.any():
        return 0.0
    weights = np.array(
        [[np.sqrt(2), 1.0, np.sqrt(2)], [1.0, 0.0, 1.0], [np.sqrt(2), 1.0, np.sqrt(2)]],
        dtype=np.float64,
    )
    # Each edge is seen from both endpoints, so the kernel is halved.
    contrib = ndimage.convolve(skel.astype(np.float64), weights / 2.0, mode="constant")
    return float(contrib[skel].sum())


def branch_and_end_points(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _neighbor_count(skel)
    branches = skel & (n >= 3)
    ends = skel & (n == 1)
    return branches, ends


def mean_width_px(mask: np.ndarray, skel: np.ndarray) -> float:
    if not skel.any():
        return 0.0
    dist = ndimage.distance_transform_edt(mask)
    return float(2.0 * dist[skel].mean())


def mean_tortuosity(skel: np.ndarray, min_length: float = 8.0) -> float:
    """Arc-chord ratio of skeleton segments between endpoints/junctions."""
    if not skel.any():
        return 0.0
    branches, _ends = branch_and_end_points(skel)
    body = skel & ~branches
    labeled, nlab = ndimage.label(body, structure=np.ones((3, 3), dtype=int))
    scores: list[float] = []
    for lab in range(1, nlab + 1):
        seg = labeled == lab
        length = skeleton_length_px(seg)
        if length < min_length:
            continue
        coords = np.argwhere(seg)
        n_local = _neighbor_count(seg)
        end_coords = np.argwhere(seg & (n_local <= 1))
        if len(end_coords) < 2:
            # Degenerate blob — use bounding extrema.
            if len(coords) < 2:
                continue
            end_coords = np.array([coords[0], coords[-1]])
        chord = float(np.linalg.norm(end_coords[0] - end_coords[-1]))
        if chord < 1e-6:
            continue
        scores.append(length / chord)
    return float(np.mean(scores)) if scores else 0.0


def morphology_metrics(mask: np.ndarray, fov: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool) & np.asarray(fov, dtype=bool)
    fov = np.asarray(fov, dtype=bool)
    skel = skeletonize(mask)
    branches, ends = branch_and_end_points(skel)
    fov_area = int(fov.sum()) or 1
    return {
        "vessel_density": float(mask.sum() / fov_area),
        "length_px": skeleton_length_px(skel),
        "mean_width_px": mean_width_px(mask, skel),
        "n_branch_points": float(branches.sum()),
        "n_end_points": float(ends.sum()),
        "mean_tortuosity": mean_tortuosity(skel),
        "skeleton_pixels": float(skel.sum()),
    }
