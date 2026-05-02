"""
=============================================================
STEP 6: NOISE-AUGMENTED FINE-TUNING (Option B)
=============================================================
Research question: Does training on noisy data make the LLM
more robust to ASR noise at test time?

NOISE AUGMENTATION STRATEGY:
  We mix clean + noisy training data at WER=10%, 20%, 30%.
  Why these levels?
  - WER=10-30% represents realistic real-world ASR quality
    (commercial ASR systems typically achieve 5-15% WER on
    clear speech, rising to 20-40% on accented/noisy speech)
  - Training on the full range (not just one WER level)
    teaches the model to generalize across noise intensities
  - WER=50% is too extreme to train on — it degrades
    label name tokens themselves, corrupting supervision

  Mix ratio: 40% clean + 20% WER10 + 20% WER20 + 20% WER30
  (keeps clean data dominant to preserve base performance)

After training, run 5_noise_robustness.py again pointing to
this adapter to get the new noise robustness curve.

Run: python 6_noise_augmented_finetune.py
=============================================================
"""

import json
import re
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForSeq2Seq,
    Trainer, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME   = "C:/Users/mojua/Desktop/Research/qwen2.5-3b"
DATA_DIR     = Path("./research_data")
OUTPUT_DIR   = Path("./results/llm_noise_augmented")
RESULT_DIR   = Path("./results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH     = 550
MAX_NEW_TOKENS = 12
BATCH_SIZE     = 8
GRAD_ACCUM     = 2
NUM_EPOCHS     = 2
LR             = 5e-5
LORA_R         = 32
LORA_ALPHA     = 64
LORA_DROPOUT   = 0.1
INFER_BATCH    = 16
RANDOM_SEED    = 42

# Noise mix: proportions of each WER level in training data
# 40% clean + 20% WER10 + 20% WER20 + 20% WER30
NOISE_MIX = {0: 0.40, 10: 0.20, 20: 0.20, 30: 0.20}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

with open(DATA_DIR / "class_mapping.json") as f:
    mapping     = json.load(f)
    class_to_id = mapping["class_to_id"]
    id_to_class = {int(k): v for k, v in mapping["id_to_class"].items()}
    classes     = mapping["classes"]

NUM_CLASSES = len(classes)
print(f"Classes: {NUM_CLASSES} | Train: {len(train_df)} | Val: {len(val_df)}")

# ─────────────────────────────────────────────
# NOISE INJECTION (same as 5_noise_robustness.py)
# ─────────────────────────────────────────────
all_words = []
for text in train_df["transcript"].fillna("").tolist():
    all_words.extend(text.lower().split())
VOCAB = list(set(all_words))

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
    "need": ["deed", "feed", "read"], "make": ["take", "lake", "wake"],
    "check": ["neck", "deck", "peck"], "call": ["tall", "ball", "hall"],
    "find": ["mind", "kind", "bind"], "show": ["so", "slow", "sow"],
    "tell": ["sell", "fell", "bell"], "when": ["then", "men", "ten"],
    "how": ["now", "cow", "wow"], "music": ["mystic", "museum", "muse"],
    "time": ["dime", "lime", "crime"], "alarm": ["arm", "farm", "calm"],
    "weather": ["whether", "feather", "leather"],
    "remind": ["refined", "behind", "bind"],
    "cancel": ["channel", "handle", "tunnel"],
    "volume": ["valve", "value", "volley"],
    "news": ["knew", "use", "views"], "recipe": ["receipt", "recite"],
    "schedule": ["school", "skill", "shell"],
    "temperature": ["temporal", "temper", "temperate"],
}

def inject_asr_noise(text, wer_percent):
    if wer_percent == 0:
        return text
    words = text.split()
    if not words:
        return text
    n_errors = max(1, round(len(words) * wer_percent / 100))
    n_sub    = round(n_errors * 0.60)
    n_del    = round(n_errors * 0.25)
    n_ins    = n_errors - n_sub - n_del
    result   = words.copy()

    for idx in random.sample(range(len(result)), min(n_sub, len(result))):
        word = result[idx].lower()
        if word in ASR_CONFUSIONS and random.random() < 0.7:
            result[idx] = random.choice(ASR_CONFUSIONS[word])
        elif len(word) > 3:
            chars = list(word)
            op    = random.choice(["swap", "delete", "replace"])
            pos   = random.randint(0, len(chars) - 1)
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

    for idx in sorted(random.sample(range(len(result)), min(n_del, len(result))), reverse=True):
        result.pop(idx)

    for _ in range(n_ins):
        if VOCAB:
            result.insert(random.randint(0, len(result)), random.choice(VOCAB[:500]))

    return " ".join(result) if result else text

# ─────────────────────────────────────────────
# BUILD NOISE-AUGMENTED TRAINING SET
# ─────────────────────────────────────────────
print("\nBuilding noise-augmented training set...")
print(f"Mix: {NOISE_MIX}")

augmented_parts = []
total_target = len(train_df)

for wer, fraction in NOISE_MIX.items():
    n_samples = int(total_target * fraction)
    subset    = train_df.sample(n=min(n_samples, len(train_df)),
                                replace=(n_samples > len(train_df)),
                                random_state=RANDOM_SEED + wer)
    if wer == 0:
        augmented_parts.append(subset.copy())
        print(f"  WER={wer:2d}% (clean):  {len(subset)} samples")
    else:
        noisy_subset = subset.copy()
        noisy_subset["transcript"] = [
            inject_asr_noise(t, wer)
            for t in tqdm(subset["transcript"].fillna("").tolist(),
                          desc=f"  Injecting WER={wer}%", leave=False)
        ]
        augmented_parts.append(noisy_subset)
        print(f"  WER={wer:2d}% (noisy):  {len(noisy_subset)} samples")

aug_train_df = pd.concat(augmented_parts).sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Total augmented training samples: {len(aug_train_df)}")

# Save augmented dataset for reproducibility
aug_train_df.to_csv(OUTPUT_DIR / "augmented_train.csv", index=False)
print(f"✅ Saved augmented training set to {OUTPUT_DIR}/augmented_train.csv")

# ─────────────────────────────────────────────
# OVERSAMPLING (same 2x cap as v3)
# ─────────────────────────────────────────────
print("\nApplying capped oversampling to augmented set...")
target    = int(aug_train_df["label"].value_counts().median() * 2)
parts = []
for label, group in aug_train_df.groupby("label"):
    if len(group) < target:
        parts.append(group.sample(target, replace=True, random_state=RANDOM_SEED))
    else:
        parts.append(group)
aug_train_df = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"After oversampling: {len(aug_train_df)} samples")

# ─────────────────────────────────────────────
# PROMPT (same as v3 — label names, class list kept)
# ─────────────────────────────────────────────
CLASS_LIST = "\n".join(classes)
SYSTEM_MSG = (
    f"You are a spoken language intent classifier.\n"
    f"Classify the utterance into exactly one of these {NUM_CLASSES} intents:\n"
    f"{CLASS_LIST}\n"
    f"Reply with ONLY the exact intent name."
)

def build_prompt(utterance, label=None):
    p = (f"<|im_start|>system\n{SYSTEM_MSG}\n<|im_end|>\n"
         f"<|im_start|>user\n{utterance}\n<|im_end|>\n"
         f"<|im_start|>assistant\n")
    if label is not None:
        p += f"{id_to_class[label]}<|im_end|>"
    return p

# ─────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token       = tokenizer.eos_token
tokenizer.padding_side    = "right"
tokenizer.truncation_side = "left"

sys_tokens = tokenizer(SYSTEM_MSG, return_tensors="pt")["input_ids"].shape[1]
print(f"System message tokens: {sys_tokens} / {MAX_LENGTH}")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
print("\nLoading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
)
model.config.use_cache = False
model.enable_input_require_grads()
print(f"✅ Model loaded: {MODEL_NAME}")

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=LORA_DROPOUT, bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# TOKENIZE
# ─────────────────────────────────────────────
def tokenize_sample(example):
    prompt_only = build_prompt(example["transcript"], label=None)
    full_prompt = build_prompt(example["transcript"], label=example["label"])
    prompt_ids  = tokenizer(prompt_only, truncation=True,
                            max_length=MAX_LENGTH-4, padding=False)["input_ids"]
    full_ids    = tokenizer(full_prompt,  truncation=True,
                            max_length=MAX_LENGTH, padding=False)["input_ids"]
    prompt_len  = min(len(prompt_ids), len(full_ids) - 1)
    labels      = [-100] * prompt_len + full_ids[prompt_len:]
    if all(l == -100 for l in labels):
        labels = [-100] * (len(full_ids) - 1) + [full_ids[-1]]
    return {"input_ids": full_ids, "attention_mask": [1]*len(full_ids), "labels": labels}

print("\nTokenizing datasets...")
train_hf  = Dataset.from_pandas(aug_train_df[["transcript","label"]])
val_hf    = Dataset.from_pandas(val_df[["transcript","label"]])
train_tok = train_hf.map(tokenize_sample, remove_columns=["transcript","label"])
val_tok   = val_hf.map(tokenize_sample,   remove_columns=["transcript","label"])
print(f"✅ Tokenized: train={len(train_tok)}, val={len(val_tok)}")

# Verification
non_masked = [l for l in train_tok[0]["labels"] if l != -100]
decoded    = tokenizer.decode(non_masked)
print(f"\n✅ VERIFICATION: '{decoded}'")
if not any(c.lower() in decoded.lower() or decoded.lower() in c.lower() for c in classes):
    print("❌ Label name not found — check tokenization!"); exit(1)
print("   ✅ Label name verified — safe to train")

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    eval_strategy="epoch", save_strategy="epoch",
    learning_rate=LR,
    bf16=True, fp16=False,
    logging_steps=50, optim="paged_adamw_8bit",
    lr_scheduler_type="cosine", warmup_steps=200,
    weight_decay=0.01, max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none", save_total_limit=2,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_tok, eval_dataset=val_tok,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, pad_to_multiple_of=8, label_pad_token_id=-100),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\n" + "="*60)
print("TRAINING WITH NOISE AUGMENTATION...")
print("="*60)
trainer.train()

model.save_pretrained(str(OUTPUT_DIR / "adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "adapter"))
print(f"✅ Noise-augmented adapter saved to {OUTPUT_DIR}/adapter")

# ─────────────────────────────────────────────
# QUICK EVAL ON CLEAN TEST SET
# ─────────────────────────────────────────────
print("\nRunning quick evaluation on clean test set...")
model.eval()
tokenizer.padding_side = "left"

classes_norm = {c.lower().strip(): class_to_id[c] for c in classes}

def match_label(text):
    t = text.replace("<|im_end|>","").strip().lower()
    if t in classes_norm: return classes_norm[t]
    for c_norm, c_id in classes_norm.items():
        if c_norm in t: return c_id
    for c_norm, c_id in classes_norm.items():
        if t in c_norm and len(t) > 3: return c_id
    return 0

def predict_batch(texts):
    prompts = [build_prompt(t) for t in texts]
    inputs  = tokenizer(prompts, return_tensors="pt", truncation=True,
                        max_length=MAX_LENGTH, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
    preds = []
    for i, out in enumerate(outputs):
        resp = tokenizer.decode(out[len(inputs["input_ids"][i]):], skip_special_tokens=True)
        preds.append(match_label(resp))
    return preds

y_true = test_df["label"].tolist()
y_pred = []
for i in tqdm(range(0, len(test_df), INFER_BATCH), desc="Evaluating"):
    y_pred.extend(predict_batch(test_df["transcript"].tolist()[i:i+INFER_BATCH]))

acc      = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
w_f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"\n{'='*50}")
print(f"NOISE-AUGMENTED MODEL — CLEAN TEST SET")
print(f"{'='*50}")
print(f"Accuracy    : {acc:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {w_f1:.4f}")

report = classification_report(y_true, y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=[id_to_class[i] for i in range(NUM_CLASSES)],
    digits=4, zero_division=0)
print(report)

with open(RESULT_DIR / "noise_augmented_clean_eval.txt", "w") as f:
    f.write(f"Noise-Augmented Model | Clean Test Set\n")
    f.write(f"Acc={acc:.4f} MacroF1={macro_f1:.4f} WeightedF1={w_f1:.4f}\n\n{report}")

print(f"\n✅ Results saved.")
print(f"\n📌 NEXT STEP:")
print(f"   Edit 5_noise_robustness.py — change LLM_ADAPTER_PATH to:")
print(f"   LLM_ADAPTER_PATH = '{OUTPUT_DIR}/adapter'")
print(f"   Then run: python 5_noise_robustness.py")
print(f"   This gives you the noise robustness curve for the augmented model.")
print(f"   Compare its curve against the original v3 adapter curve in your paper.")