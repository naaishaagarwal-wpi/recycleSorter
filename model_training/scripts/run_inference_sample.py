#!/usr/bin/env python3
"""Run the TFLite binary classifier on a random sample of images.

This script samples images from `dataset/binary_dataset/*/*`, runs the
TFLite model, and prints for each image:
 - image path
 - ground truth (folder name)
 - predicted label and probability scores

Usage:
    python scripts/run_inference_sample.py --model models/binary_classifier.tflite --data dataset/binary_dataset --num 10

If `--show` is passed the script will open each image in the default image viewer.
"""
import argparse
import os
import random
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


def load_labels_from_file(labels_path):
    labels = {}
    if not os.path.exists(labels_path):
        return None
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # expected format: "0: Recyclable"
            if ':' in line:
                idx, name = line.split(':', 1)
                try:
                    labels[int(idx.strip())] = name.strip()
                except ValueError:
                    continue
    return labels


def infer_labels_from_dirs(data_dir):
    # fallback: use subdirectory names under data_dir as labels, sorted
    dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    if not dirs:
        return None
    names = sorted([d.name for d in dirs])
    return {i: names[i] for i in range(len(names))}


def preprocess_image(img_path, input_size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((input_size[1], input_size[2]))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def run_inference(tflite_path, images, labels_map=None, show=False):
    if not os.path.exists(tflite_path):
        print(f"TFLite model not found at: {tflite_path}")
        return

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input shape e.g. [1, 224, 224, 3]
    input_shape = input_details[0]['shape']

    for img_path in images:
        gt = Path(img_path).parent.name
        input_data = preprocess_image(img_path, input_shape)

        # If model expects uint8, convert
        input_dtype_name = np.dtype(input_details[0]['dtype']).name
        if input_dtype_name == 'uint8':
            input_data = (input_data * 255).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # If output is logits (float), extract probabilities directly
        output_dtype_name = np.dtype(output_details[0]['dtype']).name
        if output_dtype_name.startswith('float'):
            probs = output_data[0]
        else:
            try:
                probs = np.squeeze(output_data)
            except Exception:
                probs = output_data[0]

        # normalize if needed
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 0:
            probs = np.array([probs])
        if probs.sum() != 0 and not np.isclose(probs.sum(), 1.0):
            # softmax
            exp = np.exp(probs - np.max(probs))
            probs = exp / exp.sum()

        pred_idx = int(np.argmax(probs))
        pred_label = labels_map.get(pred_idx, str(pred_idx)) if labels_map else str(pred_idx)

        print("---")
        print(f"Image: {img_path}")
        print(f"Ground truth folder: {gt}")
        print(f"Predicted: {pred_label} (index {pred_idx})")
        print("Probabilities:")
        for i, p in enumerate(probs.tolist()):
            name = labels_map.get(i, str(i)) if labels_map else str(i)
            print(f"  {i} ({name}): {p:.4f}")

        if show:
            Image.open(img_path).show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/binary_classifier.tflite', help='Path to TFLite model')
    parser.add_argument('--labels', default=None, help='Optional labels file (created by training script)')
    parser.add_argument('--data', default='dataset/binary_dataset', help='Binary dataset root')
    parser.add_argument('--num', type=int, default=10, help='Number of random images to sample')
    parser.add_argument('--show', action='store_true', help='Open each image in default viewer')
    args = parser.parse_args()

    # Gather image files
    patterns = [os.path.join(args.data, '*', '*')]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    # Filter images
    files = [f for f in files if os.path.isfile(f) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        print(f"No images found under {args.data}. Please prepare the binary dataset first.")
        return

    sampled = random.sample(files, min(args.num, len(files)))

    # Load labels
    labels_map = None
    if args.labels:
        labels_map = load_labels_from_file(args.labels)
    else:
        # try to find labels next to model
        labels_try = args.model.replace('.tflite', '_labels.txt')
        if os.path.exists(labels_try):
            labels_map = load_labels_from_file(labels_try)

    if labels_map is None:
        labels_map = infer_labels_from_dirs(args.data)
        if labels_map is None:
            print("Could not determine labels mapping. Predictions will use indices.")

    run_inference(args.model, sampled, labels_map=labels_map, show=args.show)


if __name__ == '__main__':
    main()
