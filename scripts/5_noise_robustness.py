"""
=============================================================
STEP 5: ASR NOISE ROBUSTNESS EXPERIMENT
=============================================================
The core research experiment: How does classification accuracy
of TF-IDF, BERT, and QLoRA LLM degrade as ASR Word Error Rate
increases from 0% to 50%?

This script:
  1. Injects controlled ASR-like noise at WER = 0,10,20,30,40,50%
  2. Evaluates ALL saved models (TF-IDF, BERT, LLM) on each noise level
  3. Saves a complete results table + plots for the paper

NOISE TYPES SIMULATED (realistic ASR errors):
  - Word substitution with phonetically/visually similar words
  - Word deletion (words dropped by ASR)
  - Word insertion (hallucinated words by ASR)
  Ratio: 60% substitution, 25% deletion, 15% insertion
  (matches real ASR error distribution from literature)

Install: pip install nlpaug

Run: python 5_noise_robustness.py
=============================================================
"""

import json
import re
import random
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = Path("./research_data")
RESULT_DIR  = Path("./results")
NOISE_DIR   = Path("./results/noise_experiment")
NOISE_DIR.mkdir(parents=True, exist_ok=True)

WER_LEVELS  = [0, 10, 20, 30, 40, 50]   # percentage

LLM_MODEL_PATH    = "C:/Users/mojua/Desktop/Research/qwen2.5-3b"
# LLM_ADAPTER_PATH  = "./results/llm_finetuned_v3/adapter"
LLM_ADAPTER_PATH = "./results/llm_noise_augmented/adapter"
BERT_MODEL_PATH   = "./results/bert_final"

INFER_BATCH = 16
MAX_LENGTH  = 550
MAX_NEW_TOKENS = 12
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# LOAD DATA & CLASS MAPPING
# ─────────────────────────────────────────────
print("Loading data...")
test_df = pd.read_csv(DATA_DIR / "test.csv")
train_df = pd.read_csv(DATA_DIR / "train.csv")

with open(DATA_DIR / "class_mapping.json") as f:
    mapping     = json.load(f)
    class_to_id = mapping["class_to_id"]
    id_to_class = {int(k): v for k, v in mapping["id_to_class"].items()}
    classes     = mapping["classes"]

NUM_CLASSES = len(classes)
y_true = test_df["label"].tolist()
print(f"Test samples: {len(test_df)} | Classes: {NUM_CLASSES}")

# ─────────────────────────────────────────────
# ASR NOISE INJECTION
# Realistic ASR error simulation without external models.
# Uses character-level perturbations for substitution,
# word deletion, and word insertion from vocabulary.
# ─────────────────────────────────────────────

# Build vocabulary from training data for realistic insertions
all_words = []
for text in train_df["transcript"].fillna("").tolist():
    all_words.extend(text.lower().split())
VOCAB = list(set(all_words))

# Common ASR confusion pairs (phonetically similar)
ASR_CONFUSIONS = {
    "the": ["a", "that", "their"], "a": ["the", "an", "i"],
    "to": ["too", "two", "the"], "and": ["an", "in", "end"],
    "is": ["it", "its", "as"], "in": ["on", "an", "and"],
    "for": ["four", "fur", "far"], "are": ["or", "our", "err"],
    "was": ["has", "as", "had"], "have": ["had", "gave", "has"],
    "not": ["note", "no", "got"], "can": ["cannot", "could", "can't"],
    "set": ["get", "let", "net"], "play": ["plate", "clay", "lay"],
    "add": ["at", "had", "and"], "what": ["that", "when", "where"],
    "turn": ["burn", "learn", "earn"], "on": ["in", "an", "one"],
    "off": ["of", "if", "out"], "my": ["by", "me", "may"],
    "the": ["a", "da", "de"], "need": ["deed", "feed", "read"],
    "make": ["take", "lake", "wake"], "check": ["neck", "deck", "check"],
    "call": ["tall", "ball", "hall"], "find": ["mind", "kind", "bind"],
    "show": ["so", "slow", "sow"], "tell": ["sell", "fell", "bell"],
    "when": ["then", "men", "ten"], "how": ["now", "cow", "wow"],
    "music": ["mystic", "museum", "muse"], "time": ["dime", "lime", "crime"],
    "weather": ["whether", "feather", "leather"],
    "alarm": ["arm", "farm", "calm"], "remind": ["refined", "behind", "bind"],
    "cancel": ["channel", "cancel", "handle"], "schedule": ["school", "skill"],
    "temperature": ["temporal", "temper"], "volume": ["valve", "value"],
    "news": ["knew", "use", "views"], "recipe": ["receipt", "recite"],
}

def inject_asr_noise(text, wer_percent):
    """
    Inject ASR-realistic noise at a given WER level.
    WER = (substitutions + deletions + insertions) / total_words

    Error type distribution (from ASR literature):
      60% substitution, 25% deletion, 15% insertion
    """
    if wer_percent == 0:
        return text

    words = text.split()
    if len(words) == 0:
        return text

    n_words    = len(words)
    n_errors   = max(1, round(n_words * wer_percent / 100))
    n_sub      = round(n_errors * 0.60)
    n_del      = round(n_errors * 0.25)
    n_ins      = n_errors - n_sub - n_del

    result = words.copy()

    # Substitutions — replace word with ASR confusion or char-perturbed version
    sub_indices = random.sample(range(len(result)), min(n_sub, len(result)))
    for idx in sub_indices:
        word = result[idx].lower()
        if word in ASR_CONFUSIONS and random.random() < 0.7:
            result[idx] = random.choice(ASR_CONFUSIONS[word])
        elif len(word) > 3:
            # Character-level perturbation (swap, delete, substitute)
            op = random.choice(["swap", "delete", "replace"])
            chars = list(word)
            pos = random.randint(0, len(chars) - 1)
            if op == "swap" and len(chars) > 1:
                i = random.randint(0, len(chars) - 2)
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif op == "delete":
                chars.pop(pos)
            else:
                chars[pos] = random.choice("aeioutnsr")
            result[idx] = "".join(chars)
        else:
            result[idx] = random.choice(VOCAB[:200]) if VOCAB else word

    # Deletions — remove words (ASR drops them)
    del_indices = sorted(
        random.sample(range(len(result)), min(n_del, len(result))),
        reverse=True
    )
    for idx in del_indices:
        result.pop(idx)

    # Insertions — insert random words (ASR hallucinations)
    for _ in range(n_ins):
        if VOCAB:
            ins_pos = random.randint(0, len(result))
            result.insert(ins_pos, random.choice(VOCAB[:500]))

    return " ".join(result) if result else text

def create_noisy_dataset(texts, wer_percent, seed=42):
    """Create a deterministic noisy version of the test set."""
    random.seed(seed)
    return [inject_asr_noise(t, wer_percent) for t in texts]

# ─────────────────────────────────────────────
# VERIFY NOISE INJECTION QUALITY
# ─────────────────────────────────────────────
print("\nVerifying noise injection...")
sample_text = "set an alarm for seven in the morning"
print(f"Original : {sample_text}")
for wer in WER_LEVELS:
    random.seed(42)
    noisy = inject_asr_noise(sample_text, wer)
    print(f"WER={wer:2d}% : {noisy}")

# Save all noisy test sets for reproducibility
print("\nGenerating and saving all noisy test sets...")
noisy_datasets = {}
utterances = test_df["transcript"].fillna("").tolist()
for wer in WER_LEVELS:
    noisy_texts = create_noisy_dataset(utterances, wer)
    noisy_datasets[wer] = noisy_texts
    # Save to CSV for full reproducibility (reviewers can inspect)
    noisy_df = pd.DataFrame({"transcript": noisy_texts, "label": y_true})
    noisy_df.to_csv(NOISE_DIR / f"test_wer{wer}.csv", index=False)
    print(f"  WER={wer}%: saved {len(noisy_texts)} samples")

# ─────────────────────────────────────────────
# ALL RESULTS STORAGE
# ─────────────────────────────────────────────
all_results = {}   # {model_name: {wer: {acc, macro_f1, weighted_f1}}}

# ─────────────────────────────────────────────
# MODEL 1: TF-IDF + SVM (retrain on clean, test on noisy)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATING: TF-IDF + Linear SVM")
print("="*60)

from sklearn.svm import LinearSVC
X_train = train_df["transcript"].fillna("").tolist()
y_train = train_df["label"].tolist()

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 3),
                        sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000, random_state=42)
svm.fit(X_train_tfidf, y_train)

svm_results = {}
for wer in WER_LEVELS:
    X_test_noisy = noisy_datasets[wer]
    X_test_tfidf = tfidf.transform(X_test_noisy)
    y_pred = svm.predict(X_test_tfidf)
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    w_f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    svm_results[wer] = {"accuracy": round(acc,4), "macro_f1": round(macro_f1,4), "weighted_f1": round(w_f1,4)}
    print(f"  WER={wer:2d}%  Acc={acc:.4f}  MacroF1={macro_f1:.4f}")

    # Save per-WER classification report
    report = classification_report(y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[id_to_class[i] for i in range(NUM_CLASSES)],
        digits=4, zero_division=0)
    with open(NOISE_DIR / f"svm_report_wer{wer}.txt", "w") as f:
        f.write(f"TF-IDF+SVM | WER={wer}%\nAcc={acc:.4f} MacroF1={macro_f1:.4f}\n\n{report}")

all_results["TF-IDF + SVM"] = svm_results

# ─────────────────────────────────────────────
# MODEL 2: TF-IDF + Logistic Regression
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATING: TF-IDF + Logistic Regression")
print("="*60)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced",
                        random_state=42, solver="lbfgs")
lr.fit(X_train_tfidf, y_train)

lr_results = {}
for wer in WER_LEVELS:
    X_test_tfidf = tfidf.transform(noisy_datasets[wer])
    y_pred = lr.predict(X_test_tfidf)
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    w_f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    lr_results[wer] = {"accuracy": round(acc,4), "macro_f1": round(macro_f1,4), "weighted_f1": round(w_f1,4)}
    print(f"  WER={wer:2d}%  Acc={acc:.4f}  MacroF1={macro_f1:.4f}")

    report = classification_report(y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[id_to_class[i] for i in range(NUM_CLASSES)],
        digits=4, zero_division=0)
    with open(NOISE_DIR / f"lr_report_wer{wer}.txt", "w") as f:
        f.write(f"TF-IDF+LR | WER={wer}%\nAcc={acc:.4f} MacroF1={macro_f1:.4f}\n\n{report}")

all_results["TF-IDF + LR"] = lr_results

# ─────────────────────────────────────────────
# MODEL 3: Fine-tuned BERT
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATING: Fine-tuned BERT-base")
print("="*60)

try:
    from transformers import AutoTokenizer as BertTok, AutoModelForSequenceClassification
    from datasets import Dataset as HFDataset
    import torch

    bert_tok   = BertTok.from_pretrained(BERT_MODEL_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)

    def bert_predict(texts, batch_size=64):
        all_preds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT inference"):
            batch = texts[i:i+batch_size]
            enc = bert_tok(batch, truncation=True, padding=True,
                           max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = bert_model(**enc).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        return all_preds

    bert_results = {}
    for wer in WER_LEVELS:
        y_pred   = bert_predict(noisy_datasets[wer])
        acc      = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        w_f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        bert_results[wer] = {"accuracy": round(acc,4), "macro_f1": round(macro_f1,4), "weighted_f1": round(w_f1,4)}
        print(f"  WER={wer:2d}%  Acc={acc:.4f}  MacroF1={macro_f1:.4f}")

        report = classification_report(y_true, y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=[id_to_class[i] for i in range(NUM_CLASSES)],
            digits=4, zero_division=0)
        with open(NOISE_DIR / f"bert_report_wer{wer}.txt", "w") as f:
            f.write(f"BERT | WER={wer}%\nAcc={acc:.4f} MacroF1={macro_f1:.4f}\n\n{report}")

    all_results["BERT-base"] = bert_results
    del bert_model
    torch.cuda.empty_cache()

except Exception as e:
    print(f"⚠️  BERT eval failed: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# MODEL 4: QLoRA Fine-tuned LLM
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATING: Qwen2.5-3B QLoRA (label-name output)")
print("="*60)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(LLM_ADAPTER_PATH, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    llm_model = PeftModel.from_pretrained(base_model, LLM_ADAPTER_PATH)
    llm_model.eval()
    print(f"✅ LLM loaded from {LLM_ADAPTER_PATH}")

    # Same prompt as training
    CLASS_LIST = "\n".join(classes)
    SYSTEM_MSG = (
        f"You are a spoken language intent classifier.\n"
        f"Classify the utterance into exactly one of these {NUM_CLASSES} intents:\n"
        f"{CLASS_LIST}\n"
        f"Reply with ONLY the exact intent name."
    )

    def build_prompt(utterance):
        return (
            f"<|im_start|>system\n{SYSTEM_MSG}\n<|im_end|>\n"
            f"<|im_start|>user\n{utterance}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    classes_norm = {c.lower().strip(): class_to_id[c] for c in classes}

    def match_label(text):
        t = text.replace("<|im_end|>", "").strip().lower()
        if t in classes_norm:
            return classes_norm[t]
        for c_norm, c_id in classes_norm.items():
            if c_norm in t:
                return c_id
        for c_norm, c_id in classes_norm.items():
            if t in c_norm and len(t) > 3:
                return c_id
        return 0

    def llm_predict_batch(texts):
        prompts = [build_prompt(t) for t in texts]
        inputs  = tokenizer(prompts, return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH, padding=True).to(llm_model.device)
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        preds = []
        for i, out in enumerate(outputs):
            new_toks = out[len(inputs["input_ids"][i]):]
            response = tokenizer.decode(new_toks, skip_special_tokens=True)
            preds.append(match_label(response))
        return preds

    llm_results = {}
    for wer in WER_LEVELS:
        noisy_texts = noisy_datasets[wer]
        y_pred = []
        for i in tqdm(range(0, len(noisy_texts), INFER_BATCH),
                      desc=f"LLM WER={wer}%"):
            batch = noisy_texts[i:i+INFER_BATCH]
            y_pred.extend(llm_predict_batch(batch))

        acc      = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        w_f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        llm_results[wer] = {"accuracy": round(acc,4), "macro_f1": round(macro_f1,4), "weighted_f1": round(w_f1,4)}
        print(f"  WER={wer:2d}%  Acc={acc:.4f}  MacroF1={macro_f1:.4f}")

        report = classification_report(y_true, y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=[id_to_class[i] for i in range(NUM_CLASSES)],
            digits=4, zero_division=0)
        with open(NOISE_DIR / f"llm_report_wer{wer}.txt", "w") as f:
            f.write(f"Qwen2.5-3B QLoRA | WER={wer}%\nAcc={acc:.4f} MacroF1={macro_f1:.4f}\n\n{report}")

    all_results["Qwen2.5-3B QLoRA"] = llm_results

except Exception as e:
    print(f"⚠️  LLM eval failed: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# SAVE ALL RESULTS
# ─────────────────────────────────────────────
with open(NOISE_DIR / "all_noise_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ All results saved to {NOISE_DIR}/all_noise_results.json")

# ─────────────────────────────────────────────
# BUILD SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("NOISE ROBUSTNESS RESULTS — MACRO F1 BY WER LEVEL")
print("="*70)

rows = []
for model_name, wer_dict in all_results.items():
    row = {"Model": model_name}
    for wer in WER_LEVELS:
        if wer in wer_dict:
            row[f"WER={wer}%"] = wer_dict[wer]["macro_f1"]
    # Degradation = drop from WER=0% to WER=50%
    if 0 in wer_dict and 50 in wer_dict:
        row["Δ (0→50%)"] = round(wer_dict[0]["macro_f1"] - wer_dict[50]["macro_f1"], 4)
    rows.append(row)

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(NOISE_DIR / "summary_macro_f1.csv", index=False)

# Also save accuracy table
acc_rows = []
for model_name, wer_dict in all_results.items():
    row = {"Model": model_name}
    for wer in WER_LEVELS:
        if wer in wer_dict:
            row[f"WER={wer}%"] = wer_dict[wer]["accuracy"]
    if 0 in wer_dict and 50 in wer_dict:
        row["Δ (0→50%)"] = round(wer_dict[0]["accuracy"] - wer_dict[50]["accuracy"], 4)
    acc_rows.append(row)

acc_df = pd.DataFrame(acc_rows)
acc_df.to_csv(NOISE_DIR / "summary_accuracy.csv", index=False)

# ─────────────────────────────────────────────
# GENERATE PAPER-QUALITY PLOTS
# ─────────────────────────────────────────────
colors  = {"TF-IDF + SVM": "#e74c3c", "TF-IDF + LR": "#e67e22",
           "BERT-base": "#3498db", "Qwen2.5-3B QLoRA": "#2ecc71"}
markers = {"TF-IDF + SVM": "s", "TF-IDF + LR": "^",
           "BERT-base": "D", "Qwen2.5-3B QLoRA": "o"}

# Plot 1: Macro F1 vs WER
fig, ax = plt.subplots(figsize=(9, 6))
for model_name, wer_dict in all_results.items():
    wers   = [w for w in WER_LEVELS if w in wer_dict]
    f1s    = [wer_dict[w]["macro_f1"] for w in wers]
    c = colors.get(model_name, "gray")
    m = markers.get(model_name, "o")
    ax.plot(wers, f1s, marker=m, linewidth=2.5, markersize=8,
            label=model_name, color=c)

ax.set_xlabel("ASR Word Error Rate (%)", fontsize=13)
ax.set_ylabel("Macro F1 Score", fontsize=13)
ax.set_title("Classification Robustness under ASR Noise\n(SLURP Dataset, 66 Intent Classes)", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.set_xticks(WER_LEVELS)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig(NOISE_DIR / "macro_f1_vs_wer.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Plot saved: macro_f1_vs_wer.png")

# Plot 2: Accuracy vs WER
fig, ax = plt.subplots(figsize=(9, 6))
for model_name, wer_dict in all_results.items():
    wers = [w for w in WER_LEVELS if w in wer_dict]
    accs = [wer_dict[w]["accuracy"] for w in wers]
    c = colors.get(model_name, "gray")
    m = markers.get(model_name, "o")
    ax.plot(wers, accs, marker=m, linewidth=2.5, markersize=8,
            label=model_name, color=c)

ax.set_xlabel("ASR Word Error Rate (%)", fontsize=13)
ax.set_ylabel("Accuracy", fontsize=13)
ax.set_title("Accuracy under ASR Noise\n(SLURP Dataset, 66 Intent Classes)", fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(WER_LEVELS)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig(NOISE_DIR / "accuracy_vs_wer.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Plot saved: accuracy_vs_wer.png")

# Plot 3: Relative degradation (normalized to WER=0%)
fig, ax = plt.subplots(figsize=(9, 6))
for model_name, wer_dict in all_results.items():
    wers     = [w for w in WER_LEVELS if w in wer_dict]
    baseline = wer_dict[0]["macro_f1"] if 0 in wer_dict else 1.0
    rel      = [wer_dict[w]["macro_f1"] / baseline * 100 for w in wers]
    c = colors.get(model_name, "gray")
    m = markers.get(model_name, "o")
    ax.plot(wers, rel, marker=m, linewidth=2.5, markersize=8,
            label=model_name, color=c)

ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Baseline (WER=0%)")
ax.set_xlabel("ASR Word Error Rate (%)", fontsize=13)
ax.set_ylabel("Relative Macro F1 (% of clean performance)", fontsize=12)
ax.set_title("Relative Performance Degradation under ASR Noise", fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(WER_LEVELS)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig(NOISE_DIR / "relative_degradation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Plot saved: relative_degradation.png")

# ─────────────────────────────────────────────
# GENERATE LaTeX TABLE FOR PAPER
# ─────────────────────────────────────────────
latex = []
latex.append(r"\begin{table}[h]")
latex.append(r"\centering")
latex.append(r"\caption{Macro F1 scores under varying ASR noise levels (WER) on SLURP dataset (66 intent classes). $\Delta$ denotes the absolute drop from WER=0\% to WER=50\%.}")
latex.append(r"\label{tab:noise_results}")
cols = "l" + "c" * len(WER_LEVELS) + "c"
latex.append(r"\begin{tabular}{" + cols + "}")
latex.append(r"\hline")
header = "Model & " + " & ".join([f"WER={w}\\%" for w in WER_LEVELS]) + r" & $\Delta$ \\"
latex.append(header)
latex.append(r"\hline")

for model_name, wer_dict in all_results.items():
    vals = []
    for wer in WER_LEVELS:
        if wer in wer_dict:
            v = wer_dict[wer]["macro_f1"]
            vals.append(f"{v:.4f}")
        else:
            vals.append("—")
    if 0 in wer_dict and 50 in wer_dict:
        delta = wer_dict[0]["macro_f1"] - wer_dict[50]["macro_f1"]
        vals.append(f"{delta:.4f}")
    else:
        vals.append("—")
    latex.append(f"{model_name} & " + " & ".join(vals) + r" \\")

latex.append(r"\hline")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

with open(NOISE_DIR / "noise_results_table.tex", "w") as f:
    f.write("\n".join(latex))
print("✅ LaTeX table saved: noise_results_table.tex")

# ─────────────────────────────────────────────
# FINAL SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("EXPERIMENT COMPLETE — OUTPUT FILES:")
print(f"  📊 Plots     : {NOISE_DIR}/macro_f1_vs_wer.png")
print(f"               : {NOISE_DIR}/accuracy_vs_wer.png")
print(f"               : {NOISE_DIR}/relative_degradation.png")
print(f"  📋 Tables    : {NOISE_DIR}/summary_macro_f1.csv")
print(f"               : {NOISE_DIR}/summary_accuracy.csv")
print(f"               : {NOISE_DIR}/noise_results_table.tex  (paste into paper)")
print(f"  📁 Reports   : {NOISE_DIR}/[model]_report_wer[N].txt (per model per WER)")
print(f"  🗃  Raw JSON  : {NOISE_DIR}/all_noise_results.json")
print(f"  📂 Noisy data: {NOISE_DIR}/test_wer[N].csv (for reproducibility)")
print("="*70)