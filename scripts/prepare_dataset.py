import os
import shutil
import random
from pathlib import Path

def prepare_dataset(data_root="dataset/binary_dataset", output_root="dataset/processed", split_ratio=0.8):
    categories = ["Not_Recyclable", "Recyclable"]
    os.makedirs(output_root, exist_ok=True)

    for cat in categories:
        files = list(Path(os.path.join(data_root, cat)).glob("*"))
        random.shuffle(files)

        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for phase, group in [("train", train_files), ("test", test_files)]:
            dest_dir = Path(output_root) / phase / cat
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in group:
                shutil.copy(f, dest_dir / f.name)

    print(f"✅ Dataset prepared in {output_root}/ with {split_ratio*100:.0f}% train split.")

if __name__ == "__main__":
    prepare_dataset()
