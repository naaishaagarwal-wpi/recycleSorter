"""Create a binary dataset (Recyclable / Not_Recyclable) from a labelled dataset.

This script is Windows-friendly and repo-relative by default. It copies images from
multiple source category folders into two target folders under the repo's
`dataset/binary_dataset` directory.

Usage (from repository root):
    python dataset\dataset_creation\createBinaryWindows.py

Optional arguments:
    --src DIR   : source labelled dataset root (default: dataset/labelled_dataset)
    --dst DIR   : destination binary dataset root (default: dataset/binary_dataset)
    --recyclable : comma-separated list of categories considered recyclable
    --notrecyc   : comma-separated list of categories considered not recyclable

The script is careful about filename collisions and will append a counter when needed.
"""
from pathlib import Path
import shutil
import argparse
import sys


DEFAULT_RECYCLABLE = [
    "paper",
    "cardboard",
    "metal",
    "green-glass",
    "brown-glass",
    "white-glass",
]

DEFAULT_NOT_RECYCLABLE = [
    "biological",
    "clothes",
    "shoes",
    "battery",
    "trash",
]


def copy_images(src_root: Path, src_folders, dst_folder: Path):
    dst_folder.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped_missing = []
    for folder in src_folders:
        folder_path = src_root / folder
        if not folder_path.exists():
            skipped_missing.append(folder)
            continue
        for p in folder_path.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
                continue

            # target path
            target = dst_folder / p.name
            # avoid collisions by appending a counter if necessary
            if target.exists():
                stem = p.stem
                suffix = p.suffix
                i = 1
                while True:
                    candidate = dst_folder / f"{stem}_{i}{suffix}"
                    if not candidate.exists():
                        target = candidate
                        break
                    i += 1

            shutil.copy2(p, target)
            count += 1

    return count, skipped_missing


def main(argv=None):
    argv = argv or sys.argv[1:]
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Create binary dataset from labelled folders")
    parser.add_argument("--src", type=Path, default=repo_root / "dataset" / "labelled_dataset",
                        help="source labelled dataset root (default: dataset/labelled_dataset)")
    parser.add_argument("--dst", type=Path, default=repo_root / "dataset" / "binary_dataset",
                        help="destination binary dataset root (default: dataset/binary_dataset)")
    parser.add_argument("--recyclable", type=str,
                        help="comma-separated recyclable categories (overrides defaults)")
    parser.add_argument("--notrecyc", type=str,
                        help="comma-separated not-recyclable categories (overrides defaults)")

    args = parser.parse_args(argv)

    src_root = args.src
    dst_root = args.dst

    recyclable = DEFAULT_RECYCLABLE if not args.recyclable else [s.strip() for s in args.recyclable.split(",")]
    not_recyclable = DEFAULT_NOT_RECYCLABLE if not args.notrecyc else [s.strip() for s in args.notrecyc.split(",")]

    # create target folders with the exact names expected by other scripts
    recyclable_dst = dst_root / "Recyclable"
    notrec_dst = dst_root / "Not_Recyclable"

    print(f"Source labelled root: {src_root}")
    print(f"Destination binary root: {dst_root}")
    print(f"Recyclable source folders: {recyclable}")
    print(f"Not recyclable source folders: {not_recyclable}")

    if not src_root.exists():
        print(f"ERROR: source root does not exist: {src_root}")
        return 2

    total = 0
    c1, missing1 = copy_images(src_root, recyclable, recyclable_dst)
    c2, missing2 = copy_images(src_root, not_recyclable, notrec_dst)
    total = c1 + c2

    print(f"Copied {c1} images to {recyclable_dst}")
    print(f"Copied {c2} images to {notrec_dst}")
    print(f"Total copied: {total}")

    if missing1 or missing2:
        print("Warning: some source folders were missing:")
        for m in (missing1 + missing2):
            print("  -", m)

    print("Done! Binary dataset is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
