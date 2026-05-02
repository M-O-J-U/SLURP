"""
=============================================================
STEP 3: LLM FINE-TUNING WITH QLoRA — SLURP Intent Classification
Qwen / 66 intent classes  |  RTX 4070 Super 12 GB
=============================================================

SPEED FIXES APPLIED (vs previous version):
  ✅ MAX_LENGTH reduced 768 → 550  (system=469, utterance avg=10 → fits perfectly)
  ✅ BATCH_SIZE reduced 8 → 4      (stable on 12 GB with 550-token sequences)
  ✅ GRAD_ACCUM increased 2 → 4    (keeps effective batch = 16, same as before)
  ✅ Batched inference (batch=32)   (was one-by-one — 20-30x faster evaluation)
  ✅ NUM_EPOCHS reduced 4 → 3      (EarlyStoppingCallback handles overfitting)
  ✅ DataCollatorForSeq2Seq padding (pads to batch-max, not global-max)
  ✅ torch.compile() disabled       (causes issues with 4-bit on Windows)

TRAINING TIME ESTIMATE (RTX 4070 Super):
  ~5-7 hours total (vs ~18-19 hours before)

Run: python 3_finetune_llm.py
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
# CONFIG  —  change only MODEL_NAME if needed
# ─────────────────────────────────────────────
MODEL_NAME  = "C:/Users/mojua/Desktop/Research/qwen2.5-3b"   # your local path
DATA_DIR    = Path("./research_data")
OUTPUT_DIR  = Path("./results/llm_finetuned")
RESULT_DIR  = Path("./results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── KEY SPEED CHANGE: 550 instead of 768 ──────────────────
# System message = ~469 tokens.  SLURP utterances = avg ~10 tokens.
# 469 + 10 + ~15 (chat template overhead) = ~494.  550 gives a safe buffer.
# BERT used MAX_LEN=64 on the utterance alone — we're being consistent:
# both BERT and LLM see the full utterance; LLM just has the system prompt too.
MAX_LENGTH     = 550

MAX_NEW_TOKENS = 8    # only need 1-2 tokens for a digit answer
BATCH_SIZE     = 2    # safe for 12 GB with 550-token sequences
GRAD_ACCUM     = 8    # effective batch = 16
NUM_EPOCHS     = 2    # EarlyStopping will cut it shorter if needed
LR             = 5e-5
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
INFER_BATCH    = 32   # batched inference — much faster than one-by-one
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
# PROMPT TEMPLATE
# ─────────────────────────────────────────────
CLASS_LIST = "\n".join([f"{i}: {c}" for i, c in enumerate(classes)])

SYSTEM_MSG = (
    f"You are a spoken language intent classifier.\n"
    f"Classify the utterance into one of these {NUM_CLASSES} intents:\n"
    f"{CLASS_LIST}\n"
    f"Reply with ONLY the integer index (0-{NUM_CLASSES-1})."
)

def build_prompt(utterance, label=None):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_MSG}\n<|im_end|>\n"
        f"<|im_start|>user\n{utterance}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if label is not None:
        prompt += f"{label}<|im_end|>"
    return prompt

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ─────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token       = tokenizer.eos_token
tokenizer.padding_side    = "right"
tokenizer.truncation_side = "left"   # preserve answer at end during truncation

# Verify system message fits comfortably
system_tokens = tokenizer(SYSTEM_MSG, return_tensors="pt")["input_ids"].shape[1]
print(f"System message tokens: {system_tokens} / {MAX_LENGTH}")
if system_tokens > MAX_LENGTH - 30:
    print("⚠️  WARNING: system message leaves very little room for the utterance!")
else:
    print(f"✅ Utterances have {MAX_LENGTH - system_tokens} tokens of room — sufficient.")

# ─────────────────────────────────────────────
# LOAD MODEL WITH 4-BIT QUANTIZATION
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
# APPLY LoRA
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
# Loss is computed ONLY on answer tokens (the class digit).
# ─────────────────────────────────────────────
def tokenize_sample(example):
    prompt_only = build_prompt(example["transcript"], label=None)
    full_prompt = build_prompt(example["transcript"], label=example["label"])

    # Use MAX_LENGTH-2 for prompt to always leave room for the answer digit
    prompt_ids = tokenizer(
        prompt_only, truncation=True,
        max_length=MAX_LENGTH - 2, padding=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_prompt, truncation=True,
        max_length=MAX_LENGTH, padding=False,
    )["input_ids"]

    prompt_len = len(prompt_ids)

    # Safety: ensure answer tokens exist
    if prompt_len >= len(full_ids):
        prompt_len = len(full_ids) - 1

    # Mask prompt tokens from loss
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # Final safety — if somehow all masked, force last token supervised
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
# CRITICAL VERIFICATION — stop if digit is missing
# ─────────────────────────────────────────────
sample_labels = train_tok[0]["labels"]
non_masked    = [l for l in sample_labels if l != -100]
decoded       = tokenizer.decode(non_masked)
print(f"\n✅ VERIFICATION:")
print(f"   Non-masked label tokens : {non_masked}")
print(f"   Decoded answer          : '{decoded}'")
if not any(c.isdigit() for c in decoded):
    print("❌ ERROR: No digit in decoded answer — fix tokenization before proceeding!")
    exit(1)
else:
    print("   ✅ Digit found — training is safe to start")

# Also print sequence length statistics for transparency
lengths = [len(x["input_ids"]) for x in train_tok]
print(f"\n   Sequence length stats:")
print(f"   Mean={np.mean(lengths):.0f} | Max={np.max(lengths)} | "
      f"Min={np.min(lengths)} | >MAX_LENGTH: {sum(l==MAX_LENGTH for l in lengths)}")

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
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    bf16=True,
    fp16=False,
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

# DataCollatorForSeq2Seq pads to the LONGEST sequence in each batch,
# not to MAX_LENGTH globally — this alone saves significant compute
# when most sequences are shorter than the maximum.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model,
    padding=True,            # pad to longest in batch, not to MAX_LENGTH
    pad_to_multiple_of=8,
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
# BATCHED INFERENCE  (replaces slow one-by-one loop)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATING ON TEST SET (batched inference)...")
print("="*60)

model.eval()

def predict_batch(utterances):
    """Predict a batch of utterances in a single GPU forward pass."""
    prompts = [build_prompt(u) for u in utterances]

    # Tokenize the whole batch at once, left-pad for generation
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens for each item in batch
    preds = []
    for i, out in enumerate(outputs):
        new_tokens = out[len(inputs["input_ids"][i]):]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        match      = re.search(r"\b(\d+)\b", response)
        if match:
            pred = int(match.group(1))
            preds.append(pred if 0 <= pred < NUM_CLASSES else 0)
        else:
            preds.append(0)

    # Reset padding side back to right for any future training
    tokenizer.padding_side = "right"
    return preds

print(f"Running batched inference (batch_size={INFER_BATCH})...")
y_true = test_df["label"].tolist()
y_pred = []

utterances = test_df["transcript"].fillna("").tolist()
for i in tqdm(range(0, len(utterances), INFER_BATCH), desc="Predicting batches"):
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
cm = confusion_matrix(y_true, y_pred)

print(f"\n{'='*50}")
print(f"MODEL: {MODEL_NAME} (QLoRA Fine-tuned)")
print(f"{'='*50}")
print(f"Accuracy    : {acc:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")
print(f"\n{report}")

# Confusion matrix plot
plt.figure(figsize=(22, 20))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=[id_to_class[i] for i in range(NUM_CLASSES)],
            yticklabels=[id_to_class[i] for i in range(NUM_CLASSES)])
plt.title(f"Confusion Matrix — {MODEL_NAME} QLoRA")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks(rotation=90, fontsize=6); plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_llm_finetuned.png", dpi=100)
plt.close()
print("✅ Confusion matrix saved")

# Save all results
llm_results = {
    "model":        MODEL_NAME,
    "method":       "QLoRA Fine-tuning",
    "dataset":      "SLURP (EMNLP 2020)",
    "task":         "intent classification (66 classes)",
    "accuracy":     round(acc, 4),
    "macro_f1":     round(macro_f1, 4),
    "weighted_f1":  round(weighted_f1, 4),
    "lora_config":  {
        "r": LORA_R, "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT, "epochs": NUM_EPOCHS, "lr": LR,
        "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
    },
}
with open(RESULT_DIR / "llm_results.json", "w") as f:
    json.dump(llm_results, f, indent=2)

with open(RESULT_DIR / "llm_classification_report.txt", "w") as f:
    f.write(f"Model: {MODEL_NAME} (QLoRA)\n")
    f.write(f"Dataset: SLURP (EMNLP 2020)\n")
    f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n{report}")

print(f"\n✅ All results saved to {RESULT_DIR}/")
print("📌 NEXT: python 4_compare_results.py")