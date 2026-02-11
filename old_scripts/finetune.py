from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted"),
    }

def finetune(
    train_csv="dataset/train_captions.csv",
    test_csv="data/test_captions.csv",
    model_name="distilbert-base-uncased",
    output_dir="models/recycle-classifier"
):
    dataset = load_dataset("csv", data_files={"train": train_csv, "test": test_csv})
    labels = sorted(set(dataset["train"]["label"]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        enc = tokenizer(examples["caption"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = [label2id[l] for l in examples["label"]]
        return enc

    dataset = dataset.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Model saved to {output_dir}")

if __name__ == "__main__":
    finetune()
