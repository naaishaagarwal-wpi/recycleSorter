import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """Create a small CNN model for binary classification."""
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
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_binary_model(data_dir='dataset/processed', model_save_path='models/binary_classifier.tflite'):
    """Train a binary classification model and save as TFLite."""
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Please run createBinary.py first.")
        return
    
    # Data generators
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=10,  # Small number for quick training
        validation_data=validation_generator
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save TFLite model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved as {model_save_path}")
    
    # Also save labels
    class_indices = train_generator.class_indices
    labels = {v: k for k, v in class_indices.items()}
    labels_file = model_save_path.replace('.tflite', '_labels.txt')
    with open(labels_file, 'w') as f:
        for i in range(len(labels)):
            f.write(f"{i}: {labels[i]}\n")
    print(f"Labels saved as {labels_file}")

if __name__ == "__main__":
    train_binary_model()