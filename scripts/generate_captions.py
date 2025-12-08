import os
import csv
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import argparse
import sys

# Suppress transformers warnings for cleaner output (optional)
import warnings
warnings.filterwarnings("ignore", message=".*You have modified the pretrained model.*")


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        print(f"✓ CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("⚠ CUDA not available. Using CPU (slower).")
        return -1


def load_pipeline(model_name: str, device: int):
    """Load image-to-text pipeline with error handling."""
    try:
        print(f"Loading model: {model_name}")
        from transformers import pipeline
        pipe = pipeline("image-to-text", model=model_name, device=device)
        print(f"✓ Model loaded successfully")
        return pipe
    except Exception as e:
        print(f"✗ Failed to load model {model_name}: {e}")
        print("  Check internet connection and disk space for model downloads.")
        raise


def validate_source(data_root: Path):
    """Check if source directory exists and contains images."""
    if not data_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {data_root}")
    
    subdirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No category subfolders found in {data_root}")
    
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    total_images = 0
    for subdir in subdirs:
        count = sum(1 for f in subdir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts)
        total_images += count
        if count == 0:
            print(f"  Warning: {subdir.name} contains no image files")
    
    if total_images == 0:
        raise ValueError(f"No image files found in {data_root}")
    
    print(f"✓ Found {total_images} images across {len(subdirs)} category folders")
    return total_images


def generate_captions(data_root: Path, output_csv: Path, model_name: str, device: int):
    """Generate captions for images in data_root and save to CSV."""
    print(f"\n{'='*60}")
    print(f"Generating captions from: {data_root}")
    print(f"Output CSV: {output_csv}")
    print(f"{'='*60}")
    
    # Validate source
    try:
        total_images = validate_source(data_root)
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Validation failed: {e}")
        return 1
    
    # Load model
    try:
        caption_pipe = load_pipeline(model_name, device)
    except Exception:
        return 1
    
    rows = []
    error_count = 0
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    
    for label_dir in sorted(data_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        
        # Collect image files
        image_files = [f for f in label_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_exts]
        
        if not image_files:
            print(f"  {label}: 0 images")
            continue
        
        print(f"\n  Processing {label} ({len(image_files)} images)...")
        for img_path in tqdm(image_files, desc=f"  {label}"):
            try:
                image = Image.open(img_path).convert("RGB")
                caption = caption_pipe(image)[0]["generated_text"].strip().lower()
                rows.append([caption, label])
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Print first 5 errors only
                    print(f"    Error processing {img_path.name}: {type(e).__name__}")
    
    if error_count > 5:
        print(f"  ... and {error_count - 5} more errors")
    
    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["caption", "label"])
            writer.writerows(rows)
        print(f"\n✓ Saved {len(rows)} captions to {output_csv}")
        if error_count > 0:
            print(f"  ({error_count} images skipped due to errors)")
        return 0 if len(rows) > 0 else 1
    except Exception as e:
        print(f"✗ Failed to write CSV: {e}")
        return 1


def main(argv=None):
    """Main entry point with CLI arguments."""
    argv = argv or sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        description="Generate image captions using BLIP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_captions.py  # default: train + test
  python scripts/generate_captions.py --train-only
  python scripts/generate_captions.py --src ./my_images --dst ./my_captions.csv --model-base
        """
    )
    
    parser.add_argument("--src-train", type=Path, default=Path("dataset") / "processed" / "train",
                        help="source train images folder (default: dataset/processed/train)")
    parser.add_argument("--src-test", type=Path, default=Path("dataset") / "processed" / "test",
                        help="source test images folder (default: dataset/processed/test)")
    parser.add_argument("--dst-train", type=Path, default=Path("dataset") / "train_captions.csv",
                        help="output train CSV (default: dataset/train_captions.csv)")
    parser.add_argument("--dst-test", type=Path, default=Path("dataset") / "test_captions.csv",
                        help="output test CSV (default: dataset/test_captions.csv)")
    parser.add_argument("--model-train", type=str, default="Salesforce/blip-image-captioning-base",
                        help="model for train split (default: blip-base, faster)")
    parser.add_argument("--model-test", type=str, default="Salesforce/blip-image-captioning-large",
                        help="model for test split (default: blip-large, slower but better)")
    parser.add_argument("--train-only", action="store_true",
                        help="only generate captions for train split")
    parser.add_argument("--test-only", action="store_true",
                        help="only generate captions for test split")
    
    args = parser.parse_args(argv)
    
    # Detect device once at start
    device = get_device()
    
    exit_code = 0
    
    # Generate train captions
    if not args.test_only:
        ret = generate_captions(args.src_train, args.dst_train, args.model_train, device)
        exit_code = max(exit_code, ret)
    
    # Generate test captions
    if not args.train_only:
        ret = generate_captions(args.src_test, args.dst_test, args.model_test, device)
        exit_code = max(exit_code, ret)
    
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✓ Caption generation complete!")
    else:
        print("✗ Caption generation completed with errors.")
    print(f"{'='*60}")
    
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
