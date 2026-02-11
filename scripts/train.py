import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    Improved CNN for binary classification.
    """
    model = keras.Sequential([
        keras.layers.Rescaling(1./255, input_shape=input_shape),

        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# ----------------------------
# Plotting utilities
# ----------------------------
def plot_metrics(history, save_path=None):
    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss', marker='o')
    plt.plot(history.history['val_loss'], label='val_loss', marker='o')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='val_acc', marker='o')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def plot_confusion(model, generator, save_path=None):
    y_true = generator.classes
    y_pred_prob = model.predict(generator, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(generator.class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

# ----------------------------
# Training function
# ----------------------------
def train_binary_model(data_dir='dataset/processed',
                       model_save_path='models/binary_classifier.tflite',
                       epochs=20,
                       batch_size=32):
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # ----------------------------
    # Data generators
    # ----------------------------
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator()
    
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary', shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary', shuffle=False
    )
    
    # ----------------------------
    # Class weights
    # ----------------------------
    class_counts = Counter(train_gen.classes)
    total = sum(class_counts.values())
    class_weight = {cls: total / (2 * count) for cls, count in class_counts.items()}
    print("Class counts:", class_counts)
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
    # Callbacks
    # ----------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    # ----------------------------
    # Training
    # ----------------------------
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks
    )
    
    # ----------------------------
    # Plot training metrics
    # ----------------------------
    plot_metrics(history, save_path='models/training_metrics.png')
    
    # ----------------------------
    # Evaluation
    # ----------------------------
    results = model.evaluate(val_gen, verbose=0)
    print("\nValidation Metrics:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
    
    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    plot_confusion(model, val_gen, save_path='models/confusion_matrix.png')
    
    # ----------------------------
    # Save TFLite
    # ----------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved to {model_save_path}")
    
    # Save labels
    labels_path = model_save_path.replace('.tflite', '_labels.txt')
    labels = {v: k for k, v in train_gen.class_indices.items()}
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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    
    train_binary_model(data_dir=args.data, model_save_path=args.model, epochs=args.epochs, batch_size=args.batch)
