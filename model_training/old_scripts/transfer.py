import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# Reproducibility
# ----------------------------
tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 5
EPOCHS_FINE = 10

# ----------------------------
# Model definition
# ----------------------------
def create_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # 🔒 freeze first

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

# ----------------------------
# Plot utilities
# ----------------------------
def plot_training(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion(model, generator):
    y_true = generator.classes
    y_pred = model.predict(generator, verbose=0)
    y_pred = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=generator.class_indices.keys())
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

# ----------------------------
# Training function
# ----------------------------
def train_binary_model(
    data_dir="dataset/processed",
    model_save_path="models/recycle_classifier.tflite"
):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = keras.preprocessing.image.ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    # ----------------------------
    # Class weights
    # ----------------------------
    counts = Counter(train_gen.classes)
    total = sum(counts.values())
    class_weight = {
        cls: total / (2 * count) for cls, count in counts.items()
    }

    print("Class counts:", counts)
    print("Class weights:", class_weight)

    # ----------------------------
    # Model
    # ----------------------------
    model, base_model = create_model()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2
        )
    ]

    # ----------------------------
    # Phase 1: Train head
    # ----------------------------
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1 
    )

    # ----------------------------
    # Phase 2: Fine-tune backbone
    # ----------------------------
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=model.metrics
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        class_weight=class_weight,
        callbacks=callbacks
    )

    # ----------------------------
    # Visuals
    # ----------------------------
    plot_training(history1)
    plot_training(history2)
    plot_confusion(model, val_gen)

    # ----------------------------
    # Evaluation
    # ----------------------------
    results = model.evaluate(val_gen, verbose=0)
    print("\nFinal validation metrics:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    # ----------------------------
    # Export to TFLite (offline-ready)
    # ----------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        f.write(tflite_model)

    print(f"\nModel saved to {model_save_path}")

    # Save labels
    labels = {v: k for k, v in train_gen.class_indices.items()}
    with open(model_save_path.replace(".tflite", "_labels.txt"), "w") as f:
        for i in labels:
            f.write(f"{i}: {labels[i]}\n")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    train_binary_model()
