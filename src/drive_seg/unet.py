"""Small 2D U-Net with Dice + BCE loss and geometric/photometric augmentation.

Ronneberger et al. (MICCAI 2015) is still the default teaching architecture
for biomedical segmentation. The model here is intentionally tiny so a CPU
demo finishes in a few minutes; the training recipe (Dice+BCE, flips/rotations)
is the same one used at DRIVE scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import Sample, load_sample
from .preprocess import prepare_gray


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum()
    den = probs.sum() + targets.sum() + eps
    return 1.0 - (num + eps) / den


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    fov: torch.Tensor | None = None,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    if fov is not None:
        weight = fov.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
        logits = logits * fov
        targets = targets * fov
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
    return (1.0 - dice_weight) * bce + dice_weight * dice_loss(logits, targets)


def _to_tensor(gray: np.ndarray, mask: np.ndarray, fov: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.ascontiguousarray(gray[None])).float()
    y = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)[None]))
    f = torch.from_numpy(np.ascontiguousarray(fov.astype(np.float32)[None]))
    return x, y, f


def augment(
    gray: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        gray, mask, fov = gray[:, ::-1], mask[:, ::-1], fov[:, ::-1]
    if rng.random() < 0.5:
        gray, mask, fov = gray[::-1], mask[::-1], fov[::-1]
    k = int(rng.integers(0, 4))
    if k:
        gray = np.rot90(gray, k)
        mask = np.rot90(mask, k)
        fov = np.rot90(fov, k)
    # Photometric jitter on the intensity image only.
    scale = float(rng.uniform(0.85, 1.15))
    shift = float(rng.uniform(-0.05, 0.05))
    gray = np.clip(gray * scale + shift, 0.0, 1.0)
    if rng.random() < 0.4:
        gray = np.clip(gray + rng.normal(0, 0.02, size=gray.shape).astype(np.float32), 0.0, 1.0)
    return gray.astype(np.float32), mask, fov


@dataclass
class VesselDataset(Dataset):
    samples: list[Sample]
    augment_on: bool = False
    seed: int = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, vessel, fov = load_sample(self.samples[index])
        gray = prepare_gray(image, invert=True)
        if self.augment_on:
            rng = np.random.default_rng(self.seed + index * 997)
            gray, vessel, fov = augment(gray, vessel, fov, rng)
        return _to_tensor(gray, vessel, fov)


@torch.no_grad()
def predict_mask(model: nn.Module, image: np.ndarray, fov: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    gray = prepare_gray(image, invert=True)
    x = torch.from_numpy(np.ascontiguousarray(gray[None, None])).float().to(device)
    logits = model(x)
    # U-Net is fully convolutional; if padding shifted the size, resample.
    if logits.shape[-2:] != gray.shape:
        logits = F.interpolate(logits, size=gray.shape, mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob >= 0.5) & fov
