import os
import csv
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import pipeline
import torch

def generate_captions(data_root="dataset/processed/train", output_csv="dataset/train_captions.csv"):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")
    print("Loading captioning model...")
    caption_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=device)

    rows = []
    for label_dir in Path(data_root).iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_path in tqdm(list(label_dir.glob("*")), desc=f"Captioning {label}"):
            try:
                image = Image.open(img_path).convert("RGB")
                caption = caption_pipe(image)[0]["generated_text"].strip().lower()
                rows.append([caption, label])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["caption", "label"])
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} captions to {output_csv}")

def generate_captions_test(data_root="dataset/processed/test", output_csv="dataset/test_captions.csv"):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")
    print("Loading captioning model...")
    caption_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)

    rows = []
    for label_dir in Path(data_root).iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_path in tqdm(list(label_dir.glob("*")), desc=f"Captioning {label}"):
            try:
                image = Image.open(img_path).convert("RGB")
                caption = caption_pipe(image)[0]["generated_text"].strip().lower()
                rows.append([caption, label])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["caption", "label"])
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} captions to {output_csv}")

if __name__ == "__main__":
    generate_captions()
    generate_captions_test()
