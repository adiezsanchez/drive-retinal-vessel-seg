"""DRIVE-style dataset I/O (official DRIVE or synthetic demo layout)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import DRIVE_DIR, SYNTHETIC_DIR

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
MASK_EXTENSIONS = (".gif", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    image_path: Path
    vessel_path: Path | None
    fov_path: Path | None


def resolve_dataset_root(prefer_drive: bool = True) -> Path:
    """Prefer official DRIVE if present; otherwise use synthetic demo data."""
    drive_ok = (DRIVE_DIR / "training" / "images").is_dir()
    synth_ok = (SYNTHETIC_DIR / "training" / "images").is_dir()
    if prefer_drive and drive_ok:
        return DRIVE_DIR
    if synth_ok:
        return SYNTHETIC_DIR
    if drive_ok:
        return DRIVE_DIR
    raise FileNotFoundError(
        "No dataset found. Run `pixi run generate-data` for the synthetic demo, "
        "or `pixi run download-drive` after obtaining DRIVE from Grand Challenge."
    )


def _first_existing(directory: Path, stem_prefixes: list[str], exts: tuple[str, ...]) -> Path | None:
    if not directory.is_dir():
        return None
    files = list(directory.iterdir())
    for prefix in stem_prefixes:
        for path in files:
            if path.suffix.lower() in exts and path.stem.lower().startswith(prefix.lower()):
                return path
    return None


def _list_images(split_dir: Path) -> list[Path]:
    image_dir = split_dir / "images"
    if not image_dir.is_dir():
        return []
    paths = [
        p
        for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]
    return paths


def _id_from_image(path: Path) -> str:
    stem = path.stem
    for suffix in ("_training", "_test", "_train"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem.split("_")[0]


def iter_samples(root: Path | None = None, split: str | None = None) -> list[Sample]:
    root = Path(root) if root is not None else resolve_dataset_root()
    splits = [split] if split else ["training", "test"]
    samples: list[Sample] = []
    for split_name in splits:
        split_dir = root / split_name
        for image_path in _list_images(split_dir):
            sample_id = _id_from_image(image_path)
            vessel = _first_existing(
                split_dir / "1st_manual",
                [f"{sample_id}_manual", f"{sample_id}_", sample_id],
                MASK_EXTENSIONS,
            )
            fov = _first_existing(
                split_dir / "mask",
                [f"{sample_id}_", sample_id],
                MASK_EXTENSIONS,
            )
            samples.append(
                Sample(
                    sample_id=sample_id,
                    split=split_name,
                    image_path=image_path,
                    vessel_path=vessel,
                    fov_path=fov,
                )
            )
    if not samples:
        raise FileNotFoundError(f"No images under {root}")
    return samples


def read_image(path: Path) -> np.ndarray:
    """Return HxWx3 uint8 RGB."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb)


def read_mask(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    """Return boolean mask, resized to ``shape_hw`` if needed."""
    if path is None:
        h, w = shape_hw
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        r = 0.48 * min(h, w)
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"))
    mask = arr > 0
    if mask.shape != shape_hw:
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (shape_hw[1], shape_hw[0]), resample=Image.NEAREST
            )
        ) > 0
    return mask


def load_sample(sample: Sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``image_rgb, vessel_mask, fov_mask``."""
    image = read_image(sample.image_path)
    hw = image.shape[:2]
    vessel = read_mask(sample.vessel_path, hw)
    fov = read_mask(sample.fov_path, hw)
    vessel &= fov
    return image, vessel, fov
