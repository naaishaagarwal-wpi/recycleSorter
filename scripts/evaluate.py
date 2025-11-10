from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model_dir="models/recycle-classifier", test_csv="data/test_captions.csv"):
    df = pd.read_csv(test_csv)
    texts, true_labels = df["caption"].tolist(), df["label"].tolist()

    classifier = pipeline("text-classification", model=model_dir)
    preds = [p["label"] for p in classifier(texts, truncation=True)]

    print("\nClassification Report:")
    print(classification_report(true_labels, preds))

    cm = confusion_matrix(true_labels, preds, labels=sorted(set(true_labels)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(set(true_labels)), yticklabels=sorted(set(true_labels)))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate()
