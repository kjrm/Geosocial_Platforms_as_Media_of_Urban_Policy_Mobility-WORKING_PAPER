"""
stage3_combine.py
=================
Stage 3: Combine Llama and second-model labels. Identify agreements and disagreements.

Accepts output from either stage2_gpt.py (label_gpt4omini) or
stage2_gemini.py (label_gemini) — detects automatically.

Input:  stage2_gemini_labels.csv  (or stage2_gpt_labels.csv)
Output: stage3_combined.csv

For each review:
  - If Llama and second model agree  -> final_label set, label_source = "llama_<model>"
  - If they disagree                 -> final_label blank, flagged for Claude tiebreak

Also prints an agreement report with Cohen's kappa.
Run this after stage2_gemini.py (or stage2_gpt.py) and before stage4_claude.py.

Setup:
  pip install pandas scikit-learn

Usage:
  python stage3_combine.py
  python stage3_combine.py --input stage2_gemini_labels.csv --output stage3_combined.csv
"""

import argparse
import logging
from typing import Optional

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def pairwise_kappa(s1: pd.Series, s2: pd.Series) -> float:
    mask = s1.notna() & s2.notna()
    if mask.sum() < 2:
        return float("nan")
    return cohen_kappa_score(s1[mask], s2[mask])


def run(args: argparse.Namespace) -> None:
    log.info(f"Loading {args.input}")
    df = pd.read_csv(args.input)

    # ── detect which second model was used ────────────────────────────────────
    second_col = None
    for candidate in ["label_gpt4omini", "label_gemini"]:
        if candidate in df.columns:
            second_col = candidate
            break

    if "label_llama3" not in df.columns or second_col is None:
        log.error("Required columns not found. Need label_llama3 and either "
                  "label_gpt4omini or label_gemini. Run stages 1 and 2 first.")
        return

    second_name = second_col.replace("label_", "")
    log.info(f"Comparing label_llama3 vs {second_col}")

    # ── compute agreement ─────────────────────────────────────────────────────
    both_labelled  = df["label_llama3"].notna() & df[second_col].notna()
    agree_mask     = both_labelled & (df["label_llama3"] == df[second_col])
    disagree_mask  = ~agree_mask

    n_total        = len(df)
    n_agree        = agree_mask.sum()
    n_disagree     = disagree_mask.sum()
    n_either_blank = (~both_labelled).sum()

    kappa  = pairwise_kappa(df["label_llama3"], df[second_col])
    interp = ("poor"          if kappa < 0.20 else
              "fair"          if kappa < 0.40 else
              "moderate"      if kappa < 0.60 else
              "substantial"   if kappa < 0.80 else
              "almost perfect")

    print("\n" + "=" * 55)
    print(f"  STAGE 3: LLAMA vs {second_name.upper()} AGREEMENT REPORT")
    print("=" * 55)
    print(f"  Total reviews:            {n_total:>6,}")
    print(f"  Both models labelled:     {both_labelled.sum():>6,}")
    print(f"  At least one blank:       {n_either_blank:>6,}")
    print(f"  Agreement:                {n_agree:>6,}  ({100*n_agree/n_total:.1f}%)")
    print(f"  Disagreement:             {n_disagree:>6,}  ({100*n_disagree/n_total:.1f}%)")
    print(f"  Cohen's kappa:            {kappa:>7.3f}  ({interp})")
    print()
    print("  Label distribution — Llama:")
    for label, count in df["label_llama3"].value_counts(dropna=False).items():
        print(f"    {str(label):<8} {count:>6,}")
    print()
    print(f"  Label distribution — {second_name}:")
    for label, count in df[second_col].value_counts(dropna=False).items():
        print(f"    {str(label):<8} {count:>6,}")
    print()
    print(f"  Disagreement breakdown (llama3 -> {second_name}):")
    disagree_df = df[disagree_mask & both_labelled]
    cross = pd.crosstab(disagree_df["label_llama3"],
                        disagree_df[second_col],
                        rownames=["llama3"], colnames=[second_name])
    print(cross.to_string())
    print("=" * 55 + "\n")

    # ── assign agreed labels ──────────────────────────────────────────────────
    df["final_label"]  = None
    df["label_source"] = None

    df.loc[agree_mask, "final_label"]  = df.loc[agree_mask, "label_llama3"]
    df.loc[agree_mask, "label_source"] = f"llama_{second_name}"
    df["needs_claude"] = disagree_mask

    df.to_csv(args.output, index=False)
    log.info(f"Output saved -> {args.output}")

    print(f"Stage 3 complete. {n_agree:,} rows resolved, "
          f"{n_disagree:,} sent to Claude tiebreak.")
    print(f"Output saved to {args.output}")
    print("Next step: python stage4_claude.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Combine Llama and second-model labels, flag disagreements.")
    parser.add_argument("--input",  default="stage2_gpt_labels.csv",
                        help="Output from stage2_gemini.py or stage2_gpt.py")
    parser.add_argument("--output", default="stage3_combined.csv")
    args = parser.parse_args()
    run(args)
