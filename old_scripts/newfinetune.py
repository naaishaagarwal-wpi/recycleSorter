# ================== MUST BE FIRST ==================
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_FILE_LOCKING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# ===================================================

import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from transformers import DebertaV2TokenizerFast
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import DebertaV2TokenizerFast


from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ------------------ STDOUT SAFETY (Windows) ------------------
sys.stdout.reconfigure(line_buffering=True)

# ------------------ Reproducibility ------------------
set_seed(42)

print("✅ Imports complete")

# ------------------ Metrics ------------------
def compute_metrics(p):
    print("📊 compute_metrics called")
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted"),
    }

# ------------------ Custom Trainer ------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        print("🧠 WeightedTrainer initialized")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ------------------ Error Analysis ------------------
def analyze_errors(trainer, dataset, id2label, output_dir):
    print("🔍 Starting error analysis")

    os.makedirs(output_dir, exist_ok=True)

    preds_output = trainer.predict(dataset)
    logits = preds_output.predictions
    preds = np.argmax(logits, axis=1)
    labels = preds_output.label_ids

    print("📉 Predictions complete")

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(id2label))
    class_names = [id2label[i] for i in tick_marks]

    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4,
    )

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n📊 Classification Report:\n")
    print(report)

    print("📁 Error analysis artifacts saved")

# ------------------ Fine-Tuning ------------------
def finetune(
    train_csv="dataset/train_captions.csv",
    test_csv="dataset/test_captions.csv",
    model_name="microsoft/deberta-v3-base",
    output_dir="models/recycle-classifier",
):

    print("🚀 finetune() started")

    os.makedirs(output_dir, exist_ok=True)

    # ---------- Load Dataset ----------
    print("📂 Loading dataset...")
    dataset = load_dataset(
        "csv",
        data_files={"train": train_csv, "test": test_csv},
    )

    print(f"✅ Dataset loaded | Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

    # ---------- Labels ----------
    labels = sorted(set(dataset["train"]["label"]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"🏷 Labels: {label2id}")

    # ---------- Tokenizer ----------
    print("🔤 Loading tokenizer...")


    tokenizer = DebertaV2TokenizerFast.from_pretrained(
        model_name,
        cache_dir="C:/hf_cache",
    )


    print("✅ Tokenizer loaded")


    def preprocess(examples):
        enc = tokenizer(examples["caption"], truncation=True)
        enc["labels"] = [label2id[l] for l in examples["label"]]
        return enc

    print("✂️ Tokenizing dataset...")
    dataset = dataset.map(preprocess, batched=True)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    print("✅ Tokenization complete")

    # ---------- Class Weights ----------
    label_counts = Counter(dataset["train"]["labels"].tolist())
    total = sum(label_counts.values())

    class_weights = torch.tensor(
        [total / label_counts[i] for i in range(len(labels))],
        dtype=torch.float,
    )
    class_weights = class_weights / class_weights.sum()

    print(f"⚖️ Class weights: {class_weights.tolist()}")

    # ---------- Model ----------
    print("🤖 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    # ---------- Training Args ----------
    training_args = TrainingArguments(
        output_dir=output_dir,

        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,

        logging_strategy="steps",
        logging_steps=10,
        disable_tqdm=False,

        learning_rate=2e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,

        weight_decay=0.01,
        label_smoothing_factor=0.1,

        load_best_model_at_end=True,
        metric_for_best_model="f1",

        report_to="none",
        max_steps=-1,  # FORCE full epochs
    )

    print("🧠 Initializing trainer...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # ---------- Train ----------
    print("🔥 TRAINING STARTED")
    trainer.train()
    print("🔥 TRAINING FINISHED")

    print("📌 Trainer state:", trainer.state)

    # ---------- Save ----------
    print("💾 Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ---------- Error Analysis ----------
    analyze_errors(
        trainer=trainer,
        dataset=dataset["test"],
        id2label=id2label,
        output_dir=output_dir,
    )

    print(f"\n✅ ALL DONE. Artifacts saved to: {output_dir}")

# ------------------ Entry Point ------------------
if __name__ == "__main__":
    print("🏁 Script entry point reached")
    finetune()
