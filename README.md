# Recycle Sorter

## Project Overview

Recycle Sorter is a mobile + machine learning project that classifies waste items as recyclable or not recyclable. It combines:

- an Android app built with Kotlin and Jetpack Compose
- an on-device TensorFlow Lite classifier
- a Python model training pipeline that prepares image data, applies transfer learning with MobileNetV2, evaluates performance, and exports a TFLite file

The goal is to help users sort waste more accurately and reward them for using the classifier through a simple points system.

## Problem Statement

Many people are uncertain whether an item belongs in recycling or trash, especially for ambiguous plastics, paper, and mixed materials. This project addresses that problem by using a lightweight image classifier to provide real-time guidance from a photo.

## Machine Learning Concepts

This project uses transfer learning to reduce training time and improve generalization on a small dataset. The model pipeline includes:

- pre-trained feature extraction with MobileNetV2 (ImageNet weights)
- a frozen convolutional base to preserve visual feature maps
- custom classifier head with global average pooling, dense layers, dropout, and sigmoid activation for binary output
- image augmentation to increase robustness to rotation, zoom, and flips
- class weighting to balance `Recyclable` and `Not Recyclable` examples
- TensorFlow Lite conversion for efficient mobile inference

## What the App Does

The Android app allows users to:

- choose or upload an image
- run local classification using a TensorFlow Lite model
- see whether the item is "Recyclable" or "Not Recyclable"
- earn points in a local Room-backed profile
- track streaks and usage across days

## Key Project Components

### Android app

- `android_app/app/src/main/java/com/example/greetingcard/MainActivity.kt`
  - app entry point
  - Jetpack Compose UI navigation between Home, Account, and classification screens
  - image picker logic and local image bitmap loading

- `android_app/app/src/main/java/com/example/greetingcard/Classifier.kt`
  - loads `binary_classifier.tflite`
  - preprocesses images with TensorImage and ResizeOp
  - runs TFLite inference
  - maps predicted scores to labels

- `android_app/app/src/main/java/com/example/greetingcard/ui/viewmodel/UserViewModel.kt`
  - manages a local user profile using Room and Kotlin coroutines
  - updates points, streaks, and total days used

- `android_app/app/build.gradle.kts`
  - configures Kotlin, Jetpack Compose, Room, TensorFlow Lite, Coil, and coroutines dependencies

### Model training and dataset

- `model_training/scripts/prepare_dataset.py`
  - prepares the binary image dataset
  - splits images into `train` and `test` folders under `model_training/dataset/processed`
  - expects category folders `Not_Recyclable` and `Recyclable` in `model_training/dataset/binary_dataset`

- `model_training/scripts/transfer-new.py`
  - performs transfer learning using MobileNetV2 as a frozen feature extractor
  - adds a custom head with global average pooling, dense layers, dropout, and a sigmoid output
  - uses data augmentation, class weights, and binary cross-entropy
  - exports the best model to `model_training/models/binary_classifier_new_big.tflite`
  - writes labels to `model_training/models/binary_classifier_new_big_labels.txt`

- `model_training/scripts/train.py` (baseline)
  - defines a smaller CNN in TensorFlow/Keras
  - augments image data using `ImageDataGenerator`
  - trains with binary cross-entropy, accuracy, precision, recall, and AUC metrics
  - saves a TFLite model to `model_training/models/binary_classifier.tflite`
  - writes labels to `model_training/models/binary_classifier_labels.txt`

- `model_training/scripts/evaluate.py`
  - evaluates the exported TFLite model on test images
  - prints a classification report
  - plots a confusion matrix

- `model_training/scripts/run_inference_sample.py`
  - runs the exported TFLite model on sample images
  - prints ground truth, prediction, and probability scores

### Dataset and model files

- `model_training/dataset/binary_dataset/Not_Recyclable/`
- `model_training/dataset/binary_dataset/Recyclable/`
- `model_training/dataset/processed/train/`
- `model_training/dataset/processed/test/`
- `model_training/models/binary_classifier_new_big.tflite`
- `model_training/models/binary_classifier_new_big_labels.txt`
- `model_training/models/binary_classifier.tflite` (CNN baseline)
- `model_training/models/binary_classifier_labels.txt` (baseline labels)

## Technologies Used

- Android / Kotlin
- Jetpack Compose UI
- Room database
- TensorFlow Lite for on-device inference
- Kotlin Coroutines
- Coil for image loading
- Python
- TensorFlow 2.15
- Keras
- NumPy
- scikit-learn
- matplotlib
- seaborn
- PIL / Pillow

## How to Run

### Android App

1. Open `android_app/` in Android Studio.
2. Build and run the `app` module.
3. Use the app to choose an image and classify it locally with the embedded TFLite model.

### Model Training

1. Create a Python environment using `model_training/environment.yml` or install packages from `model_training/requirements.txt`.
2. Prepare the dataset:
   - `cd model_training`
   - `python scripts/prepare_dataset.py`
3. Train the classifier with transfer learning:
   - `python scripts/transfer-new.py --data dataset/processed --model models/binary_classifier_new_big.tflite --epochs 12 --batch 32`
4. Evaluate the exported TFLite model:
   - `python scripts/evaluate.py`
5. Run sample inference:
   - `python scripts/run_inference_sample.py --model models/binary_classifier_new_big.tflite --data dataset/binary_dataset --num 10`

> Note: `model_training/scripts/train.py` is also available as a baseline CNN training script, but `model_training/scripts/transfer-new.py` is the transfer learning workflow used for the current model.

## Notes

- The app expects the TFLite model to be packaged as `binary_classifier.tflite` in the assets or app resources accessible via `FileUtil.loadMappedFile`.
- The model is a binary classifier for recyclability, not a multi-class waste sorter.
- The dataset split script keeps the train/test ratio at 80/20 by default.

## Useful File Paths

- Android app entry: `android_app/app/src/main/java/com/example/greetingcard/MainActivity.kt`
- Model loader: `android_app/app/src/main/java/com/example/greetingcard/Classifier.kt`
- Local user data: `android_app/app/src/main/java/com/example/greetingcard/ui/viewmodel/UserViewModel.kt`
- Android dependencies: `android_app/app/build.gradle.kts`
- Transfer learning script: `model_training/scripts/transfer-new.py`
- Baseline training script: `model_training/scripts/train.py`
- Dataset prep: `model_training/scripts/prepare_dataset.py`
- Evaluation: `model_training/scripts/evaluate.py`
- Sample inference: `model_training/scripts/run_inference_sample.py`
- Dataset root: `model_training/dataset/binary_dataset/`
- Processed dataset: `model_training/dataset/processed/`
- Saved model: `model_training/models/binary_classifier.tflite`
- Dependency list: `model_training/requirements.txt` / `model_training/environment.yml`

