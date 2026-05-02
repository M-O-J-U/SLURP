"""
=============================================================
STEP 2: BASELINE MODELS — SLURP Intent Classification
TF-IDF + LR, TF-IDF + SVM, Fine-tuned BERT-base
=============================================================
These baselines reproduce / extend the EMNLP 2020 SLURP paper
baselines. Your results will be directly comparable to theirs.

Run: python 2_baseline_models.py
=============================================================
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
warnings.filterwarnings("ignore")

DATA_DIR   = Path("./research_data")
RESULT_DIR = Path("./results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

with open(DATA_DIR / "class_mapping.json") as f:
    mapping     = json.load(f)
    id_to_class = {int(k): v for k, v in mapping["id_to_class"].items()}
    classes     = mapping["classes"]

NUM_CLASSES = len(classes)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Classes: {NUM_CLASSES}")

X_train = train_df["transcript"].fillna("").tolist()
X_test  = test_df["transcript"].fillna("").tolist()
y_train = train_df["label"].tolist()
y_test  = test_df["label"].tolist()

all_results = {}

def evaluate(model_name, y_true, y_pred):
    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1= f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[id_to_class[i] for i in range(NUM_CLASSES)],
        digits=4, zero_division=0
    )

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(report)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=[id_to_class[i] for i in range(NUM_CLASSES)],
                yticklabels=[id_to_class[i] for i in range(NUM_CLASSES)])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=90, fontsize=6); plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    safe = model_name.replace(" ", "_").replace("+", "plus")
    plt.savefig(RESULT_DIR / f"confusion_{safe}.png", dpi=100)
    plt.close()

    all_results[model_name] = {
        "model": model_name, "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),
        "report": report,
    }

# ─────────────────────────────────────────────
# MAJORITY CLASS BASELINE
# ─────────────────────────────────────────────
majority   = Counter(y_train).most_common(1)[0][0]
evaluate("Majority Class Baseline", y_test, [majority] * len(y_test))

# ─────────────────────────────────────────────
# TF-IDF VECTORIZER
# ─────────────────────────────────────────────
print("\nFitting TF-IDF (1-3 grams)...")
tfidf = TfidfVectorizer(
    max_features=50000, ngram_range=(1, 3),
    sublinear_tf=True, min_df=2,
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# ─────────────────────────────────────────────
# TF-IDF + LOGISTIC REGRESSION
# ─────────────────────────────────────────────
print("Training: TF-IDF + Logistic Regression...")
lr = LogisticRegression(
    max_iter=2000, C=5.0, class_weight="balanced",
    random_state=42, solver="lbfgs"
)
lr.fit(X_train_tfidf, y_train)
evaluate("TF-IDF + Logistic Regression", y_test, lr.predict(X_test_tfidf))

# ─────────────────────────────────────────────
# TF-IDF + LINEAR SVM
# ─────────────────────────────────────────────
print("Training: TF-IDF + Linear SVM...")
svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000, random_state=42)
svm.fit(X_train_tfidf, y_train)
evaluate("TF-IDF + Linear SVM", y_test, svm.predict(X_test_tfidf))

# ─────────────────────────────────────────────
# FINE-TUNED BERT-BASE
# ─────────────────────────────────────────────
print("\nTraining: Fine-tuned BERT-base-uncased (20-40 min on GPU)...")
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback
    )
    from datasets import Dataset

    BERT_MODEL = "bert-base-uncased"
    MAX_LEN    = 64     # SLURP utterances are short
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    BERT_LR    = 3e-5

    tok_bert = AutoTokenizer.from_pretrained(BERT_MODEL)

    def tokenize_bert(examples):
        return tok_bert(examples["transcript"],
                        padding="max_length", truncation=True, max_length=MAX_LEN)

    train_hf = Dataset.from_pandas(train_df[["transcript", "label"]])
    val_hf   = Dataset.from_pandas(val_df[["transcript",   "label"]])
    test_hf  = Dataset.from_pandas(test_df[["transcript",  "label"]])

    train_tok = train_hf.map(tokenize_bert, batched=True)
    val_tok   = val_hf.map(tokenize_bert,   batched=True)
    test_tok  = test_hf.map(tokenize_bert,  batched=True)

    model_bert = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}

    bert_args = TrainingArguments(
        output_dir="./results/bert_checkpoints",
        eval_strategy="epoch", save_strategy="epoch",
        learning_rate=BERT_LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS, weight_decay=0.01,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, fp16=torch.cuda.is_available(),
        logging_steps=100, report_to="none", save_total_limit=1,
    )

    trainer_bert = Trainer(
        model=model_bert, args=bert_args,
        train_dataset=train_tok, eval_dataset=val_tok,
        processing_class=tok_bert, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer_bert.train()

    y_pred_bert = np.argmax(trainer_bert.predict(test_tok).predictions, axis=-1)
    evaluate("Fine-tuned BERT-base", y_test, y_pred_bert.tolist())

    model_bert.save_pretrained("./results/bert_final")
    tok_bert.save_pretrained("./results/bert_final")
    print("✅ BERT saved to ./results/bert_final")

except Exception as e:
    print(f"⚠️  BERT failed: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

rows = [{"Model": r["model"], "Accuracy": r["accuracy"],
         "Macro F1": r["macro_f1"], "Weighted F1": r["weighted_f1"]}
        for r in all_results.values()]

results_df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv(RESULT_DIR / "baseline_results.csv", index=False)

with open(RESULT_DIR / "all_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "report"}
               for k, v in all_results.items()}, f, indent=2)

print(f"\n✅ Saved to {RESULT_DIR}/")
print("📌 NEXT: python 3_finetune_llm.py")