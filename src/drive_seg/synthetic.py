"""Synthetic DRIVE-like fundus images so the pipeline runs without the official set.

The official DRIVE photographs are not redistributed. These images are generated
procedurally (optic disc + branching dark vessels + circular FOV) and are only
a computational stand-in. They are *not* a clinical substitute for DRIVE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .paths import SYNTHETIC_DIR


def _draw_disk(canvas: np.ndarray, cy: float, cx: float, radius: float, value: float) -> None:
    h, w = canvas.shape
    yy, xx = np.ogrid[:h, :w]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    canvas[disk] = value


def _paint_polyline(mask: np.ndarray, points: np.ndarray, width: float) -> None:
    if len(points) < 2:
        return
    tmp = np.zeros_like(mask, dtype=bool)
    for y, x in points:
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
            tmp[iy, ix] = True
    if not tmp.any():
        return
    radius = max(width / 2.0, 0.6)
    d = ndimage.distance_transform_edt(~tmp)
    mask |= d <= radius


def _grow_tree(
    mask: np.ndarray,
    fov: np.ndarray,
    start: tuple[float, float],
    angle: float,
    length: float,
    width: float,
    depth: int,
    rng: np.random.Generator,
) -> None:
    if depth <= 0 or width < 0.7 or length < 10:
        return
    y, x = start
    n = max(int(length), 4)
    pts = []
    a = angle
    for _ in range(n):
        a += float(rng.normal(0, 0.07))
        y += float(np.sin(a))
        x += float(np.cos(a))
        iy, ix = int(round(y)), int(round(x))
        if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]):
            break
        if not fov[iy, ix]:
            break
        pts.append((y, x))
    if len(pts) < 4:
        return
    arr = np.asarray(pts, dtype=np.float32)
    _paint_polyline(mask, arr, width)
    end = (float(arr[-1, 0]), float(arr[-1, 1]))
    if rng.random() < 0.85:
        _grow_tree(mask, fov, end, a + float(rng.uniform(0.25, 0.7)), length * 0.68, width * 0.72, depth - 1, rng)
        _grow_tree(mask, fov, end, a - float(rng.uniform(0.25, 0.7)), length * 0.62, width * 0.68, depth - 1, rng)
    elif rng.random() < 0.5:
        _grow_tree(mask, fov, end, a + float(rng.uniform(-0.2, 0.2)), length * 0.7, width * 0.75, depth - 1, rng)


def _make_fov(h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = 0.47 * min(h, w)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def synthesize_fundus(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = w = size
    fov = _make_fov(h, w)
    vessels = np.zeros((h, w), dtype=bool)

    disc_y = float(rng.uniform(0.42 * h, 0.58 * h))
    disc_x = float(rng.uniform(0.28 * w, 0.40 * w))
    n_roots = int(rng.integers(4, 7))
    for i in range(n_roots):
        angle = float(i * (2 * np.pi / n_roots) + rng.uniform(-0.2, 0.2))
        # Prefer vessels leaving the disc toward the macula (right of disc).
        start = (disc_y + 6 * np.sin(angle), disc_x + 6 * np.cos(angle))
        _grow_tree(
            vessels,
            fov,
            start,
            angle,
            length=float(rng.uniform(size * 0.22, size * 0.38)),
            width=float(rng.uniform(2.8, 4.4)),
            depth=4,
            rng=rng,
        )

    # Background: reddish fundus, darker toward the macula, noisy choroid.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fundus = np.zeros((h, w, 3), dtype=np.float32)
    fundus[..., 0] = 0.62
    fundus[..., 1] = 0.22
    fundus[..., 2] = 0.10
    low = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), sigma=12)
    low = (low - low.min()) / (np.ptp(low) + 1e-6)
    fundus += 0.08 * low[..., None]
    mac_y, mac_x = disc_y, min(w * 0.68, disc_x + 0.32 * w)
    mac = np.exp(-(((yy - mac_y) ** 2 + (xx - mac_x) ** 2) / (2 * (0.08 * size) ** 2)))
    fundus -= 0.12 * mac[..., None]
    disc = np.exp(-(((yy - disc_y) ** 2 + (xx - disc_x) ** 2) / (2 * (0.045 * size) ** 2)))
    fundus += disc[..., None] * np.array([0.25, 0.22, 0.05])[None, None, :]
    fundus += rng.normal(0, 0.015, fundus.shape).astype(np.float32)

    # Dark vessels, slightly stronger in the green channel (DRIVE-like contrast).
    vsoft = ndimage.gaussian_filter(vessels.astype(np.float32), sigma=0.6)
    fundus[..., 0] -= 0.28 * vsoft
    fundus[..., 1] -= 0.38 * vsoft
    fundus[..., 2] -= 0.18 * vsoft
    fundus = np.clip(fundus, 0, 1)
    fundus[~fov] = 0
    rgb = (fundus * 255).astype(np.uint8)
    return rgb, vessels & fov, fov


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype == bool or array.dtype == np.bool_:
        Image.fromarray(array.astype(np.uint8) * 255).save(path)
    else:
        Image.fromarray(array).save(path)


def generate_synthetic_dataset(
    root: Path | None = None,
    size: int = 256,
    n_train: int = 8,
    n_test: int = 4,
    seed: int = 7,
) -> Path:
    root = Path(root) if root is not None else SYNTHETIC_DIR
    rng = np.random.default_rng(seed)
    configs = [("training", n_train, 21), ("test", n_test, 1)]
    for split, count, start_id in configs:
        for i in range(count):
            idx = start_id + i
            rgb, vessels, fov = synthesize_fundus(size, rng)
            tag = "training" if split == "training" else "test"
            _save_png(root / split / "images" / f"{idx:02d}_{tag}.png", rgb)
            _save_png(root / split / "1st_manual" / f"{idx:02d}_manual1.png", vessels)
            _save_png(root / split / "mask" / f"{idx:02d}_{tag}_mask.png", fov)
    return root
