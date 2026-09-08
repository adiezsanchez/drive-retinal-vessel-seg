#!/usr/bin/env python3
"""Generate synthetic DRIVE-layout fundus images for the demo pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drive_seg.synthetic import generate_synthetic_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    root = generate_synthetic_dataset(
        size=args.size, n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    print(f"Wrote synthetic DRIVE-layout dataset to {root}")


if __name__ == "__main__":
    main()
