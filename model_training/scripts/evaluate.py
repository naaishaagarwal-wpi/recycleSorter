import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_tflite(
    tflite_path="models/binary_classifier.tflite",
    test_dir="dataset/processed/test",
    img_size=(224, 224)
):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    X, y_true = [], []

    class_map = {}
    for idx, cls in enumerate(sorted(os.listdir(test_dir))):
        class_map[cls] = idx
        cls_dir = os.path.join(test_dir, cls)
        for fname in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, fname)
            img = image.load_img(img_path, target_size=img_size)
            img = image.img_to_array(img) / 255.0
            X.append(img)
            y_true.append(idx)

    preds = []

    for img in X:
        input_tensor = np.expand_dims(img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        prob = interpreter.get_tensor(output_details[0]["index"])[0][0]
        preds.append(int(prob > 0.5))

    print("\nClassification Report:")
    print(classification_report(y_true, preds, target_names=class_map.keys()))

    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_map.keys(),
                yticklabels=class_map.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (TFLite)")
    plt.show()

if __name__ == "__main__":
    evaluate_tflite()
    # ----------------------------
    # Plotting functions
    # ----------------------------
def plot_metrics(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss', marker='o')
    plt.plot(history.history['val_loss'], label='val_loss', marker='o')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='val_acc', marker='o')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.show()
def plot_confusion(model, generator):
    y_true = generator.classes
    y_pred_prob = model.predict(generator, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(generator.class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()