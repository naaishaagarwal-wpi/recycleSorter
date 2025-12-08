import shutil
import random
import argparse
from pathlib import Path
import sys


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def collect_image_files(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def prepare_dataset(data_root: Path, output_root: Path, split_ratio: float = 0.8, categories=None):
    output_root.mkdir(parents=True, exist_ok=True)

    # detect categories if not provided
    if categories is None:
        # all immediate subdirectories of data_root
        categories = [p.name for p in sorted(data_root.iterdir()) if p.is_dir()]

    if not categories:
        print(f"No category subfolders found in {data_root}")
        return 1

    summary = {}
    for cat in categories:
        src_cat = data_root / cat
        files = collect_image_files(src_cat)
        if not files:
            print(f"Warning: no image files found for category '{cat}' at {src_cat}")
            summary[cat] = (0, 0)
            continue

        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for phase, group in [("train", train_files), ("test", test_files)]:
            dest_dir = output_root / phase / cat
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in group:
                # copy preserving metadata
                shutil.copy2(f, dest_dir / f.name)

        summary[cat] = (len(train_files), len(test_files))

    # print summary
    total_train = sum(v[0] for v in summary.values())
    total_test = sum(v[1] for v in summary.values())
    print(f"✅ Dataset prepared in {output_root}/ with {split_ratio*100:.0f}% train split.")
    for cat, (t, s) in summary.items():
        print(f" - {cat}: train={t}, test={s}")
    print(f"Total: train={total_train}, test={total_test}")
    return 0


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(description="Prepare train/test splits from labelled image folders")
    # choose dataset/binary_dataset as default if it exists, otherwise use labelled_dataset
    default_binary = Path("dataset") / "binary_dataset"
    default_labelled = Path("dataset") / "labelled_dataset"
    default_src = default_binary if default_binary.exists() else default_labelled
    parser.add_argument("--src", type=Path, default=default_src,
                        help=f"source labelled dataset root (default: {default_src})")
    parser.add_argument("--dst", type=Path, default=Path("dataset") / "processed",
                        help="output root for processed dataset (default: dataset/processed)")
    parser.add_argument("--split", type=float, default=0.8, help="train split ratio (default: 0.8)")
    parser.add_argument("--categories", type=str,
                        help="comma-separated list of category folder names to use (overrides auto-detect)")

    args = parser.parse_args(argv)

    categories = None
    if args.categories:
        categories = [s.strip() for s in args.categories.split(",") if s.strip()]
    else:
        # if the source is a binary dataset with expected folder names, prefer that ordering
        possible_binary = ["Not_Recyclable", "Recyclable"]
        if all((args.src / p).exists() for p in possible_binary):
            categories = possible_binary

    return prepare_dataset(args.src, args.dst, split_ratio=args.split, categories=categories)


if __name__ == "__main__":
    raise SystemExit(main())
