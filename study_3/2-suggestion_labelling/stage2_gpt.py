"""
stage2_gpt.py
=============
Stage 2: Run GPT-4o-mini on all filtered reviews.

Input:  stage1_llama_labels.csv  (output of stage1_llama.py)
Output: stage2_gpt_labels.csv

Adds a label_gpt4omini column to the stage 1 output.
Run this after stage1_llama.py and before stage3_combine.py.

Setup:
  pip install openai pandas tqdm
  export OPENAI_API_KEY="sk-..."

Usage:
  python stage2_gpt.py
  python stage2_gpt.py --input stage1_llama_labels.csv --output stage2_gpt_labels.csv
  python stage2_gpt.py --test_rows 30 --max_workers 2
"""

import os
import re
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

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
VALID_LABELS  = {"YES", "NO", "MAYBE"}


def parse_label(raw: str) -> Optional[str]:
    if not raw:
        return None
    match = re.search(r"\b(YES|NO|MAYBE)\b", raw.upper())
    return match.group(1) if match else None


def call_openai(
    text: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    retries: int = 5,
    inter_request_delay: float = 1,
) -> Optional[str]:
    time.sleep(inter_request_delay)   # gentle pacing between every request
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=50,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_TEMPLATE.format(text=text)},
                ],
            )
            return parse_label(resp.choices[0].message.content)
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err:
                # Rate limit: back off much longer than a generic error
                wait = 60 * (attempt + 1)
                log.warning(f"Rate limit hit. Waiting {wait}s before retry {attempt+1}.")
            else:
                wait = 2 ** attempt
                log.warning(f"OpenAI attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set. Exiting.")
        return

    log.info(f"Loading {args.input}")
    sample = pd.read_csv(args.input)

    if "label_llama3" not in sample.columns:
        log.error("label_llama3 column not found. Run stage1_llama.py first.")
        return

    if args.test_rows:
        sample = sample.sample(n=min(args.test_rows, len(sample)),
                               random_state=42).reset_index(drop=True)
        log.info(f"TEST MODE: {len(sample)} rows")

    log.info(f"Running GPT-4o-mini on {len(sample):,} reviews ...")
    caller = ModelCaller(call_openai,
                         client=OpenAI(api_key=api_key),
                         model="gpt-4o-mini")
    sample["label_gpt4omini"] = run_model(
        sample["review_text"].tolist(), "gpt4omini", caller, args.max_workers)

    n_blank = sample["label_gpt4omini"].isna().sum()
    dist    = sample["label_gpt4omini"].value_counts(dropna=False).to_dict()
    log.info(f"Done. Blanks: {n_blank}  Distribution: {dist}")

    sample.to_csv(args.output, index=False)
    log.info(f"Output saved -> {args.output}")
    print(f"\nStage 2 complete. {len(sample):,} rows saved to {args.output}")
    print(f"Blank labels: {n_blank}")
    print("Next step: python stage3_combine.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: GPT-4o-mini labelling.")
    parser.add_argument("--input",       default="stage1_llama_labels.csv")
    parser.add_argument("--output",      default="stage2_gpt_labels.csv")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Parallel threads. Keep low (2-3) to avoid rate limits.")
    parser.add_argument("--test_rows",   type=int, default=None,
                        help="If set, run on a random sample of this many rows")
    args = parser.parse_args()
    run(args)
