"""
stage4_claude.py
================
Stage 4: Run Claude Sonnet as tiebreaker on Llama/GPT disagreements.
         Produce the final combined dataset ready for human revision.

Input:  stage3_combined.csv  (output of stage3_combine.py)
Output: final_labelled.csv          — all reviews with final labels
        final_human_review.csv      — rows still unresolved after Claude

Resolution logic:
  - Claude agrees with Llama OR GPT  -> resolved, label_source = "claude_tiebreak"
  - Claude also disagrees            -> needs_human = True
  - Claude returns blank             -> needs_human = True

Run this after stage3_combine.py.

Setup:
  pip install anthropic pandas tqdm
  export ANTHROPIC_API_KEY="sk-ant-..."

Usage:
  python stage4_claude.py
  python stage4_claude.py --input stage3_combined.csv --output final_labelled.csv
  python stage4_claude.py --test_rows 30 --max_workers 2
"""

import os
import re
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import anthropic
from tqdm import tqdm

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a research assistant analyzing customer reviews of brewpubs in Pennsylvania.

Your task is to determine if a review contains a SUGGESTION for improving the product, service, or overall experience.

Classification guidelines:
- YES:   The review explicitly suggests something the business could do better, change, or add.
         Examples: "They should have more vegetarian options", "The service could be faster"
- NO:    The review only praises or complains without suggesting improvements.
         Examples: "Great beer, loved it!", "The food was terrible"
- MAYBE: The review implies a suggestion but doesn't explicitly state it.
         Examples: "The beer selection was limited", "It was very crowded"

Respond with ONE word only: YES, NO, or MAYBE. No headers, no explanation, no punctuation."""

USER_TEMPLATE = 'Review:\n"""\n{text}\n"""'


def parse_label(raw: str) -> Optional[str]:
    if not raw:
        return None
    match = re.search(r"\b(YES|NO|MAYBE)\b", raw.upper())
    return match.group(1) if match else None


def call_anthropic(
    text: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-5",
    retries: int = 3,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=50,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_TEMPLATE.format(text=text)}
                ],
            )
            return parse_label(resp.content[0].text)
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"Anthropic attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
            time.sleep(wait)
    return None


class ModelCaller:
    def __init__(self, fn, **kwargs):
        self.fn     = fn
        self.kwargs = kwargs

    def __call__(self, text: str) -> Optional[str]:
        return self.fn(text, **self.kwargs)


def run_model(texts: list, model_name: str, caller: ModelCaller,
              max_workers: int = 5) -> list:
    labels = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(caller, t): i for i, t in enumerate(texts)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"  {model_name}", unit="review"):
            i = futures[fut]
            try:
                labels[i] = fut.result()
            except Exception as e:
                log.warning(f"Row {i} failed: {e}")
    return labels


def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set. Exiting.")
        return

    log.info(f"Loading {args.input}")
    df = pd.read_csv(args.input)

    for col in ["label_llama3", "label_gpt4omini", "needs_claude", "final_label"]:
        if col not in df.columns:
            log.error(f"Column '{col}' not found. Run stage3_combine.py first.")
            return

    # ── isolate disagreement rows ─────────────────────────────────────────────
    disagree_idx = df.index[df["needs_claude"] == True].tolist()
    disagree_df  = df.loc[disagree_idx].copy()

    if args.test_rows:
        disagree_df = disagree_df.sample(
            n=min(args.test_rows, len(disagree_df)), random_state=42)
        disagree_idx = disagree_df.index.tolist()
        log.info(f"TEST MODE: running Claude on {len(disagree_df)} disagreement rows")
    else:
        log.info(f"Running Claude on {len(disagree_df):,} disagreement rows ...")

    # ── run Claude ────────────────────────────────────────────────────────────
    caller = ModelCaller(call_anthropic,
                         client=anthropic.Anthropic(api_key=api_key),
                         model="claude-sonnet-4-5")
    claude_labels = run_model(
        disagree_df["review_text"].tolist(), "claude", caller, args.max_workers)

    df.loc[disagree_idx, "label_claude"] = claude_labels

    # ── resolve tiebreaks ─────────────────────────────────────────────────────
    claude_col  = df.loc[disagree_idx, "label_claude"]
    llama_col   = df.loc[disagree_idx, "label_llama3"]
    gpt_col     = df.loc[disagree_idx, "label_gpt4omini"]

    claude_llama_match = claude_col.notna() & (claude_col == llama_col)
    claude_gpt_match   = claude_col.notna() & (claude_col == gpt_col)
    resolved_mask      = claude_llama_match | claude_gpt_match

    resolved_idx   = disagree_df.index[resolved_mask]
    unresolved_idx = disagree_df.index[~resolved_mask]

    df.loc[resolved_idx, "final_label"]  = df.loc[resolved_idx, "label_claude"]
    df.loc[resolved_idx, "label_source"] = "claude_tiebreak"
    df["needs_human"] = df["final_label"].isna()

    # ── summary ───────────────────────────────────────────────────────────────
    n_total      = len(df)
    n_auto       = (~df["needs_human"]).sum()
    n_human      = df["needs_human"].sum()
    n_blank_cld  = df.loc[disagree_idx, "label_claude"].isna().sum()

    print("\n" + "=" * 55)
    print("  STAGE 4: FINAL CLASSIFICATION SUMMARY")
    print("=" * 55)
    print(f"  Total reviews:              {n_total:>6,}")
    print(f"  Resolved by Llama + GPT:    "
          f"{(df['label_source'] == 'llama_gpt').sum():>6,}")
    print(f"  Resolved by Claude:         {len(resolved_idx):>6,}")
    print(f"  Claude blank labels:        {n_blank_cld:>6,}")
    print(f"  Still unresolved (human):   {n_human:>6,}  "
          f"({100*n_human/n_total:.1f}%)")
    print()
    print("  Final label distribution (auto-resolved):")
    dist = df.loc[~df["needs_human"], "final_label"].value_counts()
    for label, count in dist.items():
        print(f"    {label:<8} {count:>6,}  ({100*count/n_auto:.1f}%)")
    print()
    print("  Label source breakdown:")
    for src, count in df["label_source"].value_counts(dropna=False).items():
        print(f"    {str(src):<22} {count:>6,}")
    print("=" * 55 + "\n")

    # ── save final labelled dataset ───────────────────────────────────────────
    meta_cols  = ["review_user_id", "gmap_id", "business_name",
                  "municipality", "rating", "review_text", "has_response"]
    label_cols = [c for c in ["label_llama3", "label_gpt4omini", "label_claude"]
                  if c in df.columns]
    out_cols   = (
        [c for c in meta_cols if c in df.columns]
        + label_cols
        + ["final_label", "label_source", "needs_human"]
    )

    df[out_cols].to_csv(args.output, index=False)
    log.info(f"Final dataset saved -> {args.output}")

    # ── save human review file ────────────────────────────────────────────────
    if n_human > 0:
        human_df = df.loc[df["needs_human"], out_cols].copy()
        human_df["human_label"] = ""   # blank column for annotator
        human_df.to_csv(args.human_output, index=False)
        log.info(f"Human review file ({n_human} rows) saved -> {args.human_output}")
    else:
        log.info("No rows need human review — no human review file created.")

    print(f"Done. Final dataset: {args.output}")
    if n_human > 0:
        print(f"Human review file:  {args.human_output}  ({n_human} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4: Claude tiebreaker and final combined dataset.")
    parser.add_argument("--input",        default="stage3_combined.csv")
    parser.add_argument("--output",       default="final_labelled.csv")
    parser.add_argument("--human_output", default="final_human_review.csv")
    parser.add_argument("--max_workers",  type=int, default=5)
    parser.add_argument("--test_rows",    type=int, default=None,
                        help="If set, run Claude on only this many disagreement rows")
    args = parser.parse_args()
    run(args)
