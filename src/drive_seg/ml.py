"""Pixel-wise Random Forest and linear SVM on hand-crafted features.

Each FOV pixel is a sample. Features combine intensity, local statistics,
gradients, Hessian ridge cues, and Frangi vesselness — the classic 2010s
DRIVE pipeline before fully convolutional nets took over.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage
from skimage.filters import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .classical import vesselness
from .preprocess import prepare_gray


FEATURE_NAMES = (
    "intensity",
    "local_mean_3",
    "local_std_3",
    "local_mean_7",
    "local_std_7",
    "sobel",
    "frangi",
    "hessian_ridge",
    "dist_center",
)


def _local_mean_std(gray: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    mean = ndimage.uniform_filter(gray, size=size, mode="nearest")
    mean_sq = ndimage.uniform_filter(gray**2, size=size, mode="nearest")
    var = np.clip(mean_sq - mean**2, 0, None)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _hessian_ridge(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Bright-ridge cue: λ2 ≈ 0, λ1 << 0 for a 2D bright tube after inversion."""
    blurred = ndimage.gaussian_filter(gray, sigma=sigma, mode="nearest")
    dy, dx = np.gradient(blurred)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    # Eigenvalues of [[dxx, dxy], [dyx, dyy]]
    tmp = np.sqrt(np.clip((dxx - dyy) ** 2 + 4 * dxy * dyx, 0, None))
    lam1 = 0.5 * (dxx + dyy - tmp)
    lam2 = 0.5 * (dxx + dyy + tmp)
    # Order so |λ1| >= |λ2|
    swap = np.abs(lam1) < np.abs(lam2)
    lam1, lam2 = lam1.copy(), lam2.copy()
    lam1[swap], lam2[swap] = lam2[swap], lam1[swap]
    ridge = np.clip(-lam1, 0, None) * np.exp(-np.abs(lam2) / (np.abs(lam1) + 1e-6))
    peak = float(ridge.max()) or 1.0
    return (ridge / peak).astype(np.float32)


def extract_features(image: np.ndarray) -> np.ndarray:
    """Return an HxWxC float32 feature volume."""
    gray = prepare_gray(image, invert=True)
    m3, s3 = _local_mean_std(gray, 3)
    m7, s7 = _local_mean_std(gray, 7)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    dist /= dist.max() or 1.0
    stack = np.stack(
        [
            gray,
            m3,
            s3,
            m7,
            s7,
            sobel(gray).astype(np.float32),
            vesselness(image),
            _hessian_ridge(gray),
            dist,
        ],
        axis=-1,
    )
    return stack.astype(np.float32)


def _balanced_indices(y: np.ndarray, rng: np.random.Generator, max_per_class: int) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n = min(max_per_class, len(pos), len(neg))
    if n == 0:
        return np.arange(len(y))
    return np.concatenate(
        [rng.choice(pos, size=n, replace=False), rng.choice(neg, size=n, replace=False)]
    )


def pixels_from_sample(
    image: np.ndarray,
    vessel: np.ndarray,
    fov: np.ndarray,
    rng: np.random.Generator,
    max_per_class: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    feats = extract_features(image)
    fov_idx = np.flatnonzero(fov.ravel())
    x = feats.reshape(-1, feats.shape[-1])[fov_idx]
    y = vessel.ravel().astype(np.int32)[fov_idx]
    keep = _balanced_indices(y, rng, max_per_class)
    return x[keep], y[keep]


def build_random_forest(random_state: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=4,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )


def build_svm(random_state: int = 0) -> Pipeline:
    # Linear SVM: a kernel SVM on every pixel is too slow for a teaching demo.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=4000,
                    dual=False,
                    random_state=random_state,
                ),
            ),
        ]
    )


def predict_mask(model, image: np.ndarray, fov: np.ndarray) -> np.ndarray:
    feats = extract_features(image)
    x = feats.reshape(-1, feats.shape[-1])
    pred = model.predict(x).reshape(fov.shape).astype(bool)
    pred &= fov
    return pred


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)
