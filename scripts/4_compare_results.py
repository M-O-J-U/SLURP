"""
=============================================================
STEP 4: GENERATE PAPER-READY RESULTS TABLE — SLURP
=============================================================
Combines all model results into a comparison table
formatted for your research paper with LaTeX output.

Run: python 4_compare_results.py
=============================================================
"""

import json
import pandas as pd
from pathlib import Path

RESULT_DIR = Path("./results")

all_rows = []

# Load baseline results
baseline_csv = RESULT_DIR / "baseline_results.csv"
if baseline_csv.exists():
    baselines = pd.read_csv(baseline_csv)
    for _, row in baselines.iterrows():
        all_rows.append({
            "Model":       row["Model"],
            "Method":      "Traditional / Encoder",
            "Accuracy":    row["Accuracy"],
            "Macro F1":    row["Macro F1"],
            "Weighted F1": row["Weighted F1"],
        })
else:
    print("⚠️  baseline_results.csv not found — run 2_baseline_models.py first")

# Load LLM results
llm_json = RESULT_DIR / "llm_results.json"
if llm_json.exists():
    with open(llm_json) as f:
        llm = json.load(f)
    all_rows.append({
        "Model":       llm["model"],
        "Method":      llm.get("method", "QLoRA Fine-tuning"),
        "Accuracy":    llm["accuracy"],
        "Macro F1":    llm["macro_f1"],
        "Weighted F1": llm["weighted_f1"],
    })
else:
    print("⚠️  llm_results.json not found — run 3_finetune_llm.py first")

if not all_rows:
    print("❌ No results found. Run scripts 2 and 3 first.")
    exit(1)

# Build table
results_df = pd.DataFrame(all_rows).sort_values("Macro F1", ascending=False)
results_df = results_df.reset_index(drop=True)
results_df.index += 1

print("\n" + "=" * 70)
print("FINAL RESULTS TABLE — SLURP Intent Classification")
print("=" * 70)
print(results_df.to_string())

# Save CSV
results_df.to_csv(RESULT_DIR / "final_comparison_table.csv", index=True)
print(f"\n✅ CSV saved → {RESULT_DIR}/final_comparison_table.csv")

# Save LaTeX
latex = results_df[["Model", "Accuracy", "Macro F1", "Weighted F1"]].to_latex(
    index=True, float_format="%.4f",
    caption=(
        "Comparison of models for intent classification on the SLURP dataset "
        "(EMNLP 2020). All models use the official train/devel/test splits. "
        "Macro F1 is the primary metric."
    ),
    label="tab:slurp_results",
)
with open(RESULT_DIR / "results_table.tex", "w") as f:
    f.write(latex)
print(f"✅ LaTeX saved → {RESULT_DIR}/results_table.tex")
print("\nPaste the LaTeX directly into your paper's Results section.")
print("=" * 70)