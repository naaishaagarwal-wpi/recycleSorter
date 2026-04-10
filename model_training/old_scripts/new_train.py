import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import Counter

# ----------------------------
# Reproducibility
# ----------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ----------------------------
# Model definition
# ----------------------------
def create_model(input_shape=(224, 224, 3)):
    """
    Binary CNN using sigmoid output.
    """
    model = keras.Sequential([
        keras.layers.Rescaling(1./255, input_shape=input_shape),

        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),

        # Binary output
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# ----------------------------
# Training function
# ----------------------------
def train_binary_model(
    data_dir='dataset/processed',
    model_save_path='models/binary_classifier.tflite',
    epochs=10
):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} does not exist")

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # ----------------------------
    # Data generators
    # ----------------------------
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = keras.preprocessing.image.ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    # ----------------------------
    # Class weights (CRITICAL)
    # ----------------------------
    class_counts = Counter(train_generator.classes)
    total = sum(class_counts.values())

    class_weight = {
        cls: total / (2 * count)
        for cls, count in class_counts.items()
    }

    print("Class distribution:", class_counts)
    print("Class weights:", class_weight)

    # ----------------------------
    # Model
    # ----------------------------
    model = create_model()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    # ----------------------------
    # Training
    # ----------------------------
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        class_weight=class_weight
    )

    # ----------------------------
    # Evaluation
    # ----------------------------
    results = model.evaluate(validation_generator, verbose=0)
    metrics = dict(zip(model.metrics_names, results))

    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ----------------------------
    # Export to TFLite
    # ----------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)

    print(f"\nModel saved to {model_save_path}")

    # ----------------------------
    # Save labels
    # ----------------------------
    labels = {v: k for k, v in train_generator.class_indices.items()}
    labels_path = model_save_path.replace('.tflite', '_labels.txt')

    with open(labels_path, 'w') as f:
        for i in sorted(labels):
            f.write(f"{i}: {labels[i]}\n")

    print(f"Labels saved to {labels_path}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset/processed')
    parser.add_argument('--model', default='models/binary_classifier.tflite')
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    train_binary_model(
        data_dir=args.data,
        model_save_path=args.model,
        epochs=args.epochs
    )
