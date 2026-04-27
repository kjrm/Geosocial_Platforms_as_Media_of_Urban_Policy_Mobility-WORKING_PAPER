"""
stage1_llama.py
===============
Stage 1: Run Llama 3 (via Ollama) on all filtered reviews.

Input:  PA_brewpub_reviews.csv  (or --input)
Output: stage1_llama_labels.csv

The output contains all filtered reviews with a label_llama3 column.
Run this before stage2_gpt.py.

Setup:
  pip install requests pandas tqdm
  # Install Ollama: https://ollama.com
  ollama pull llama3
  ollama serve   # keep running in a separate terminal

Usage:
  python stage1_llama.py
  python stage1_llama.py --input my_data.csv --output my_llama_labels.csv
  python stage1_llama.py --test_rows 30 --max_workers 2
"""

import re
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
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
VALID_LABELS  = {"YES", "NO", "MAYBE"}


def parse_label(raw: str) -> Optional[str]:
    if not raw:
        return None
    match = re.search(r"\b(YES|NO|MAYBE)\b", raw.upper())
    return match.group(1) if match else None


def call_ollama(
    text: str,
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    retries: int = 3,
) -> Optional[str]:
    payload = {
        "model":   model,
        "prompt":  f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(text=text)}",
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 50},
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            return parse_label(r.json().get("response", ""))
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"Ollama attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
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


def build_sample(df: pd.DataFrame, test_rows: Optional[int] = None) -> pd.DataFrame:
    sample = df[
        (df["has_response"] == True) &
        (df["rating"] < 5) &
        (df["review_text"].notna()) &
        (df["review_text"].str.strip() != "")
    ].copy().reset_index(drop=True)

    if test_rows:
        sample = sample.sample(n=min(test_rows, len(sample)),
                               random_state=42).reset_index(drop=True)
        log.info(f"TEST MODE: {len(sample)} rows")
    else:
        log.info(f"Sample size: {len(sample):,} reviews")
    return sample


def run(args: argparse.Namespace) -> None:
    log.info(f"Loading {args.input}")
    df     = pd.read_csv(args.input)
    sample = build_sample(df, test_rows=args.test_rows)

    log.info(f"Running Llama 3 (model={args.llama_model}) ...")
    caller = ModelCaller(call_ollama, model=args.llama_model, base_url=args.ollama_url)
    sample["label_llama3"] = run_model(
        sample["review_text"].tolist(), "llama3", caller, args.max_workers)

    n_blank = sample["label_llama3"].isna().sum()
    dist    = sample["label_llama3"].value_counts(dropna=False).to_dict()
    log.info(f"Done. Blanks: {n_blank}  Distribution: {dist}")

    sample.to_csv(args.output, index=False)
    log.info(f"Output saved -> {args.output}")
    print(f"\nStage 1 complete. {len(sample):,} rows saved to {args.output}")
    print(f"Blank labels: {n_blank}")
    print("Next step: python stage2_gpt.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Llama 3 labelling.")
    parser.add_argument("--input",       default="PA_brewpub_reviews.csv")
    parser.add_argument("--output",      default="stage1_llama_labels.csv")
    parser.add_argument("--llama_model", default="llama3")
    parser.add_argument("--ollama_url",  default="http://localhost:11434")
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--test_rows",   type=int, default=None,
                        help="If set, run on a random sample of this many rows")
    args = parser.parse_args()
    run(args)
