#!/usr/bin/env python3
"""Train a small U-Net with Dice+BCE on CUDA when available (CPU fallback)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drive_seg.data import iter_samples, load_sample, resolve_dataset_root
from drive_seg.device import describe_device, get_device
from drive_seg.io_pred import save_prediction
from drive_seg.paths import MODELS_DIR
from drive_seg.unet import UNet, VesselDataset, bce_dice_loss, predict_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(describe_device())

    root = resolve_dataset_root()
    train_samples = iter_samples(root, split="training")
    test_samples = iter_samples(root, split="test")
    train_ds = VesselDataset(train_samples, augment_on=True, seed=args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = UNet(in_ch=1, out_ch=1, base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for images, masks, fovs in loader:
            images = images.to(device)
            masks = masks.to(device)
            fovs = fovs.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce_dice_loss(logits, masks, fovs)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"epoch {epoch:02d}/{args.epochs}  loss={sum(losses)/len(losses):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS_DIR / "unet.pt"
    torch.save({"state_dict": model.state_dict(), "device": str(device)}, ckpt)
    print(f"Saved {ckpt}")

    model.eval()
    for sample in tqdm(test_samples, desc="unet:test"):
        image, _vessel, fov = load_sample(sample)
        pred = predict_mask(model, image, fov, device)
        save_prediction("unet", sample.sample_id, sample.split, pred)
    print("Wrote U-Net predictions.")


if __name__ == "__main__":
    main()
