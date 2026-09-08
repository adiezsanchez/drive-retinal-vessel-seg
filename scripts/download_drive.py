#!/usr/bin/env python3
"""Obtain the official DRIVE dataset (not redistributed in this repository).

DRIVE must be downloaded from Grand Challenge after registration:

    https://drive.grand-challenge.org/

This script never fetches unofficial mirrors. Pass the zip you downloaded
with ``--archive`` and it will extract into ``data/drive/`` and validate
the expected training/test layout.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE_DIR = ROOT / "data" / "drive"
GRAND_CHALLENGE = "https://drive.grand-challenge.org/"
DOWNLOAD = "https://drive.grand-challenge.org/Download/"


def _looks_like_drive(root: Path) -> bool:
    return (root / "training" / "images").is_dir() and (root / "test" / "images").is_dir()


def _find_drive_root(extracted: Path) -> Path | None:
    if _looks_like_drive(extracted):
        return extracted
    for child in extracted.rglob("training"):
        candidate = child.parent
        if _looks_like_drive(candidate):
            return candidate
    return None


def _print_instructions() -> None:
    print(
        f"""
DRIVE (Digital Retinal Images for Vessel Extraction) is a 40-image fundus
benchmark (Staal et al., IEEE TMI 2004). It is NOT bundled here.

1. Create a free account on Grand Challenge.
2. Open {GRAND_CHALLENGE}
3. Download the official training/test archives from {DOWNLOAD}
4. Re-run this script with the zip:

     pixi run download-drive -- --archive /path/to/DRIVE.zip

Expected layout after extraction:

  data/drive/training/{{images,1st_manual,mask}}
  data/drive/test/{{images,1st_manual,mask}}   # 1st_manual on test if provided

Until then, `pixi run generate-data` builds a synthetic stand-in so every
script and notebook still runs end-to-end.
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to the official DRIVE zip downloaded from Grand Challenge.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DRIVE_DIR,
        help="Extraction directory (default: data/drive).",
    )
    args = parser.parse_args()

    if args.archive is None:
        _print_instructions()
        if _looks_like_drive(args.dest):
            print(f"\nExisting DRIVE layout detected at {args.dest}")
        sys.exit(0)

    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        print(f"Archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} → {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)

    found = _find_drive_root(dest)
    if found is None:
        print(
            "Extracted files, but could not find training/images and test/images.\n"
            "Move the official DRIVE folders under data/drive/ manually.",
            file=sys.stderr,
        )
        sys.exit(2)

    if found.resolve() != dest.resolve():
        print(f"DRIVE folders found at {found}")
        print(f"Point DRIVE_DIR at that path or move them to {dest}")
    print("DRIVE layout looks valid.")
    print("The rest of the pipeline uses data/drive/ when present, else synthetic demo data.")


if __name__ == "__main__":
    main()
