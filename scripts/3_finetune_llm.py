"""
=============================================================
STEP 3: LLM FINE-TUNING — SLURP Intent Classification
Qwen2.5-3B-Instruct | 66 classes | Label-Name Output
=============================================================

KEY IMPROVEMENTS OVER v2:
  ✅ Model outputs LABEL NAMES instead of digit indices
     (more stable tokens, better semantic alignment)
  ✅ LoRA R=32, alpha=64 (higher capacity for 66 classes)
  ✅ Minority class oversampling (capped at 2x, not full balance)
  ✅ Robust inference: fuzzy label name matching
  ✅ Class list KEPT in prompt (critical for noise robustness research)
  ✅ truncation_side="left" restored (answer never truncated)
  ✅ MAX_LENGTH=550 (system prompt is ~469 tokens, need the room)
  ✅ Verification step before training
  ✅ eval_strategy (not deprecated evaluation_strategy)

WHY CLASS LIST IS KEPT:
  For the ASR noise paper, inference must work on corrupted inputs
  the model has never seen. Without the class list, the model cannot
  reliably output valid label names on out-of-distribution noisy text.
  Removing it would invalidate the noise robustness experiment.

Run: python 3_finetune_llm_v3.py
=============================================================
"""

import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
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
MODEL_NAME  = "C:/Users/mojua/Desktop/Research/qwen2.5-3b"
DATA_DIR    = Path("./research_data")
OUTPUT_DIR  = Path("./results/llm_finetuned_v3")
RESULT_DIR  = Path("./results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# System prompt is ~469 tokens — need 550 to fit utterance too
MAX_LENGTH     = 550
MAX_NEW_TOKENS = 12   # label names can be longer than a single digit
BATCH_SIZE     = 4
GRAD_ACCUM     = 4    # effective batch = 16
NUM_EPOCHS     = 3
LR             = 5e-5

# Higher LoRA capacity for 66-class problem
LORA_R         = 32
LORA_ALPHA     = 64
LORA_DROPOUT   = 0.1

INFER_BATCH    = 16   # smaller batch for generation with longer outputs
OVERSAMPLE_CAP = 2    # minority classes oversampled up to 2x majority, not full balance
RANDOM_SEED    = 42

# ─────────────────────────────────────────────
# LOAD DATA & CLASS MAPPING
# ─────────────────────────────────────────────
print("Loading data and class mapping...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

with open(DATA_DIR / "class_mapping.json") as f:
    mapping     = json.load(f)
    class_to_id = mapping["class_to_id"]
    id_to_class = {int(k): v for k, v in mapping["id_to_class"].items()}
    classes     = mapping["classes"]

NUM_CLASSES = len(classes)
print(f"Classes: {NUM_CLASSES}")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ─────────────────────────────────────────────
# MINORITY CLASS OVERSAMPLING (capped at 2x majority)
# Full balance (new code suggestion) balloons train set to 66x larger.
# 2x cap gives imbalance correction without exploding training time.
# ─────────────────────────────────────────────
print("\nApplying capped oversampling...")
counts     = train_df["label"].value_counts()
max_count  = counts.max()
cap        = int(max_count * OVERSAMPLE_CAP / counts.max())   # = 2 effectively
target     = min(max_count, int(counts.median() * OVERSAMPLE_CAP))

parts = []
for label, group in train_df.groupby("label"):
    if len(group) < target:
        sampled = group.sample(target, replace=True, random_state=RANDOM_SEED)
    else:
        sampled = group
    parts.append(sampled)

train_df = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Train size after oversampling: {len(train_df)}")
print(f"Class count range: {train_df['label'].value_counts().min()} – {train_df['label'].value_counts().max()}")

# ─────────────────────────────────────────────
# PROMPT TEMPLATE — outputs LABEL NAME, not digit
# Class list kept for noise robustness (see docstring above)
# ─────────────────────────────────────────────
CLASS_LIST = "\n".join(classes)   # just names, no index numbers

SYSTEM_MSG = (
    f"You are a spoken language intent classifier.\n"
    f"Classify the utterance into exactly one of these {NUM_CLASSES} intents:\n"
    f"{CLASS_LIST}\n"
    f"Reply with ONLY the exact intent name."
)

def build_prompt(utterance, label=None):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_MSG}\n<|im_end|>\n"
        f"<|im_start|>user\n{utterance}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if label is not None:
        prompt += f"{id_to_class[label]}<|im_end|>"
    return prompt

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ─────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token       = tokenizer.eos_token
tokenizer.padding_side    = "right"
tokenizer.truncation_side = "left"   # preserves answer at end during truncation

system_tokens = tokenizer(SYSTEM_MSG, return_tensors="pt")["input_ids"].shape[1]
print(f"System message tokens: {system_tokens} / {MAX_LENGTH}")
print(f"Room for utterance: {MAX_LENGTH - system_tokens} tokens")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("\nLoading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.enable_input_require_grads()
print(f"✅ Model loaded: {MODEL_NAME}")

# ─────────────────────────────────────────────
# APPLY LoRA (higher capacity: R=32)
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT, bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# TOKENIZE WITH PROMPT MASKING
# Loss only on label name tokens (not the prompt)
# ─────────────────────────────────────────────
def tokenize_sample(example):
    prompt_only = build_prompt(example["transcript"], label=None)
    full_prompt = build_prompt(example["transcript"], label=example["label"])

    prompt_ids = tokenizer(
        prompt_only, truncation=True,
        max_length=MAX_LENGTH - 4,   # leave room for label name tokens
        padding=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_prompt, truncation=True,
        max_length=MAX_LENGTH, padding=False,
    )["input_ids"]

    prompt_len = len(prompt_ids)
    if prompt_len >= len(full_ids):
        prompt_len = len(full_ids) - 1

    labels = [-100] * prompt_len + full_ids[prompt_len:]

    if all(l == -100 for l in labels):
        labels = [-100] * (len(full_ids) - 1) + [full_ids[-1]]

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }

print("\nTokenizing datasets...")
train_hf  = Dataset.from_pandas(train_df[["transcript", "label"]])
val_hf    = Dataset.from_pandas(val_df[["transcript", "label"]])

train_tok = train_hf.map(tokenize_sample, remove_columns=["transcript", "label"])
val_tok   = val_hf.map(tokenize_sample,   remove_columns=["transcript", "label"])
print(f"✅ Tokenized: train={len(train_tok)}, val={len(val_tok)}")

# ─────────────────────────────────────────────
# VERIFICATION — confirm label name tokens are supervised
# ─────────────────────────────────────────────
sample_labels = train_tok[0]["labels"]
non_masked    = [l for l in sample_labels if l != -100]
decoded       = tokenizer.decode(non_masked)
print(f"\n✅ VERIFICATION:")
print(f"   Non-masked tokens  : {non_masked}")
print(f"   Decoded answer     : '{decoded}'")

# Check decoded answer contains a known class name
answer_clean = decoded.replace("<|im_end|>", "").strip().lower()
matched = any(c.lower() in answer_clean or answer_clean in c.lower() for c in classes)
if not matched:
    print(f"❌ ERROR: Decoded answer '{decoded}' doesn't match any class name!")
    print(f"   Classes sample: {classes[:5]}")
    exit(1)
else:
    print(f"   ✅ Label name found in answer — training is safe to start")

lengths = [len(x["input_ids"]) for x in train_tok]
print(f"\n   Sequence length: Mean={np.mean(lengths):.0f} | "
      f"Max={np.max(lengths)} | >MAX_LENGTH: {sum(l==MAX_LENGTH for l in lengths)}")

# ─────────────────────────────────────────────
# TRAINING ARGUMENTS
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    eval_strategy="epoch",           # NOT evaluation_strategy (deprecated)
    save_strategy="epoch",
    learning_rate=LR,
    bf16=True, fp16=False,
    logging_steps=50,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=200,
    weight_decay=0.01,
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    save_total_limit=2,
    dataloader_num_workers=0,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model,
    padding=True, pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_tok, eval_dataset=val_tok,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING LLM WITH QLoRA...")
print("="*60)
trainer.train()

model.save_pretrained(str(OUTPUT_DIR / "adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "adapter"))
print(f"✅ Adapter saved to {OUTPUT_DIR}/adapter")

# ─────────────────────────────────────────────
# BATCHED INFERENCE WITH LABEL NAME MATCHING
# Fuzzy matching: handles minor formatting differences
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATING ON TEST SET...")
print("="*60)

model.eval()

# Pre-build normalized class lookup for fast matching
classes_normalized = {c.lower().strip(): class_to_id[c] for c in classes}

def match_label(text):
    """Match generated text to nearest class label name."""
    text_clean = text.replace("<|im_end|>", "").strip().lower()

    # Exact match first
    if text_clean in classes_normalized:
        return classes_normalized[text_clean]

    # Substring match: generated text contains the class name
    for c_norm, c_id in classes_normalized.items():
        if c_norm in text_clean:
            return c_id

    # Substring match: class name contains the generated text
    for c_norm, c_id in classes_normalized.items():
        if text_clean in c_norm and len(text_clean) > 3:
            return c_id

    return 0   # fallback to class 0

def predict_batch(utterances):
    prompts = [build_prompt(u) for u in utterances]

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH, padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    preds = []
    for i, out in enumerate(outputs):
        new_tokens = out[len(inputs["input_ids"][i]):]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
        preds.append(match_label(response))

    tokenizer.padding_side = "right"
    return preds

print(f"Running batched inference (batch_size={INFER_BATCH})...")
y_true     = test_df["label"].tolist()
y_pred     = []
utterances = test_df["transcript"].fillna("").tolist()

for i in tqdm(range(0, len(utterances), INFER_BATCH), desc="Predicting"):
    batch = utterances[i : i + INFER_BATCH]
    y_pred.extend(predict_batch(batch))

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
acc         = accuracy_score(y_true, y_pred)
macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
report      = classification_report(
    y_true, y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=[id_to_class[i] for i in range(NUM_CLASSES)],
    digits=4, zero_division=0
)

print(f"\n{'='*50}")
print(f"MODEL: Qwen2.5-3B QLoRA v3 (label-name output)")
print(f"{'='*50}")
print(f"Accuracy    : {acc:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")
print(f"\n{report}")

# Save
results = {
    "model": MODEL_NAME, "version": "v3_labelname",
    "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
    "weighted_f1": round(weighted_f1, 4),
    "config": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                "lr": LR, "epochs": NUM_EPOCHS, "max_length": MAX_LENGTH},
}
with open(RESULT_DIR / "llm_results_v3.json", "w") as f:
    json.dump(results, f, indent=2)

with open(RESULT_DIR / "llm_report_v3.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n\n{report}")

print(f"\n✅ Results saved to {RESULT_DIR}/")
print("📌 NEXT: python 4_compare_results.py")