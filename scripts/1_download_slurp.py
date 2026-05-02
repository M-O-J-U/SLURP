"""
=============================================================
SLURP DATASET — Download & Preparation Script (v2 FIXED)
=============================================================
SLURP: Spoken Language Understanding Resource Package
EMNLP 2020 | 72k utterances | 18 domains | 46 intents

FIX: intent field is a ClassLabel integer — decoded via
     dataset.features["intent"].names lookup table.

Run: python 1_download_slurp.py
=============================================================
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR  = Path("./research_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_SAMPLES = 30    # drop classes with fewer samples
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("Downloading SLURP from HuggingFace (cached if already done)...")
print("=" * 60)

dataset = load_dataset("qmeeus/slurp")

print(f"\nSplits: {list(dataset.keys())}")
for s, d in dataset.items():
    print(f"  {s}: {len(d)} rows")

# ─────────────────────────────────────────────
# STEP 2 — DECODE INTENT LABELS
# Intent is a ClassLabel — integer index into a name list
# e.g. 10 → "calendar_set"
# ─────────────────────────────────────────────
print("\nDecoding intent ClassLabel names...")

# Get the label name list from the train split features
intent_names = dataset["train"].features["intent"].names
print(f"Total intent classes in dataset: {len(intent_names)}")
print(f"First 10 intent names: {intent_names[:10]}")

def decode_intent(idx, names):
    """Convert integer class index to clean intent string."""
    if idx is None or idx < 0 or idx >= len(names):
        return None
    raw = names[idx]           # e.g. "10calendar_set"
    # Strip leading digits
    clean = raw.lstrip("0123456789")   # e.g. "calendar_set"
    return clean if clean else None

def get_domain(intent_str):
    """Extract domain from intent string e.g. 'calendar_set' → 'calendar'"""
    if not intent_str:
        return None
    return intent_str.split("_")[0] if "_" in intent_str else intent_str

# ─────────────────────────────────────────────
# STEP 3 — PROCESS EACH SPLIT
# ─────────────────────────────────────────────
def process_split(split_data, split_name, names):
    split_data = split_data.remove_columns(["audio"])
    records = []
    for row in tqdm(split_data, desc=f"Processing {split_name}"):
        sentence   = str(row.get("sentence", "")).strip()
        intent_idx = row.get("intent")

        if not sentence:
            continue

        intent = decode_intent(intent_idx, names)
        if intent is None:
            continue

        domain = get_domain(intent)

        records.append({
            "transcript": sentence,
            "intent":     intent,
            "domain":     domain,
        })
    return pd.DataFrame(records)

print("\nProcessing splits...")
train_df = process_split(dataset["train"], "train", intent_names)
val_df   = process_split(dataset["devel"], "devel", intent_names)
test_df  = process_split(dataset["test"],  "test",  intent_names)

print(f"\n✅ Processed:")
print(f"  train : {len(train_df)}")
print(f"  val   : {len(val_df)}")
print(f"  test  : {len(test_df)}")

# ─────────────────────────────────────────────
# STEP 4 — EXPLORE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATASET EXPLORATION")
print("=" * 60)

print(f"\nUnique intents in train : {train_df['intent'].nunique()}")
print(f"Unique domains in train : {train_df['domain'].nunique()}")

print(f"\nDomain distribution (train):")
print(train_df['domain'].value_counts())

print(f"\nIntent distribution (train) — top 20:")
print(train_df['intent'].value_counts().head(20))

print(f"\nSample utterances:")
for intent in train_df['intent'].unique()[:5]:
    sample = train_df[train_df['intent'] == intent]['transcript'].iloc[0]
    print(f"  [{intent}]: {sample}")

# ─────────────────────────────────────────────
# STEP 5 — FILTER RARE CLASSES
# Use INTENT as classification label (46 classes — harder, more novel)
# ─────────────────────────────────────────────
print(f"\nFiltering classes with < {MIN_SAMPLES} samples...")

# Count based on train only
train_counts  = train_df['intent'].value_counts()
valid_intents = train_counts[train_counts >= MIN_SAMPLES].index.tolist()

train_df = train_df[train_df['intent'].isin(valid_intents)].reset_index(drop=True)
val_df   = val_df[val_df['intent'].isin(valid_intents)].reset_index(drop=True)
test_df  = test_df[test_df['intent'].isin(valid_intents)].reset_index(drop=True)

print(f"Kept {len(valid_intents)} intents")
print(f"After filtering: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

# ─────────────────────────────────────────────
# STEP 6 — ENCODE LABELS
# ─────────────────────────────────────────────
classes     = sorted(valid_intents)
class_to_id = {c: i for i, c in enumerate(classes)}
id_to_class = {i: c for c, i in class_to_id.items()}

train_df["label"] = train_df["intent"].map(class_to_id).astype(int)
val_df["label"]   = val_df["intent"].map(class_to_id).astype(int)
test_df["label"]  = test_df["intent"].map(class_to_id).astype(int)

print(f"\nClass mapping sample (first 10):")
for i, c in list(id_to_class.items())[:10]:
    print(f"  {i}: {c}")

# ─────────────────────────────────────────────
# STEP 7 — SAVE
# ─────────────────────────────────────────────
print("\nSaving...")

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
val_df.to_csv(  OUTPUT_DIR / "val.csv",   index=False, encoding="utf-8")
test_df.to_csv( OUTPUT_DIR / "test.csv",  index=False, encoding="utf-8")

with open(OUTPUT_DIR / "class_mapping.json", "w") as f:
    json.dump({
        "class_to_id": class_to_id,
        "id_to_class": {str(k): v for k, v in id_to_class.items()},
        "classes":     classes,
        "task":        "intent",
        "num_classes": len(classes),
    }, f, indent=2)

stats = {
    "dataset":       "SLURP",
    "paper":         "EMNLP 2020 — Bastianelli et al.",
    "task":          "intent classification",
    "num_classes":   len(classes),
    "num_domains":   train_df["domain"].nunique(),
    "train_size":    len(train_df),
    "val_size":      len(val_df),
    "test_size":     len(test_df),
    "avg_utterance_len_chars": int(train_df["transcript"].str.len().mean()),
    "class_distribution_train": train_df["intent"].value_counts().to_dict(),
}
with open(OUTPUT_DIR / "dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ All saved to {OUTPUT_DIR}/")

print("\n" + "=" * 60)
print("✅ COMPLETE")
print("=" * 60)
print(f"  Intents (classes) : {len(classes)}")
print(f"  Domains           : {train_df['domain'].nunique()}")
print(f"  Train             : {len(train_df)}")
print(f"  Val               : {len(val_df)}")
print(f"  Test              : {len(test_df)}")
print(f"\n📌 NEXT: python 2_baseline_models.py")
print("=" * 60)