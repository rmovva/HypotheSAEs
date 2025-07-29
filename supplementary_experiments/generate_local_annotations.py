#!/usr/bin/env python3
"""
Generate annotations for local LLMs & OpenAI LLMs to benchmark annotation quality relative to gpt-4.1.

Example
-------
python generate_local_annotations.py --model 'Qwen/Qwen3-0.6B'
"""

import argparse
import os
import time
import json
import pickle

import numpy as np
import pandas as pd
from hypothesaes.annotate import annotate
from hypothesaes.llm_local import get_vllm_engine, is_local_model

from pathlib import Path
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

current_dir = os.getcwd()
prefix = './' if current_dir.endswith("HypotheSAEs") else '../'
DATA_PATH = os.path.join(prefix, "demo_data", "yelp-demo-val-2K.json")

CONCEPTS = [
    "uses superlative language to describe the restaurant as the best in a specific category (e.g., 'best lobster roll', 'best bakery', 'best cheesesteaks')",
    "mentions experiences of food causing illness, such as food poisoning, stomach pain, or vomiting",
    "mentions long wait times for seating, food, or service",
    "positively describes the restaurant's handling of dietary restrictions (e.g., vegan options, allergy-friendly, etc.)",
    "describes a mixed experience, where the food was good but there were issues with the service or ambiance",
]

# Root directory for all experiment outputs
ROOT = Path('/nas/ucb/rmovva/data/hypothesaes')

THINKING_OPTIONS = [True, False]

# ---------------------------------------------------------------------------
# Paths for results
# ---------------------------------------------------------------------------
MODELS = [
    "HuggingFaceTB/SmolLM3-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B-AWQ",
    #
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
]
TIMING_RESULTS_PATH = ROOT / "local_llm_experiments/results/annotation_timing_results.jsonl"
ANNOTATIONS_PATH = ROOT / "local_llm_experiments/results/yelp_benchmark_annotations.pkl"


def build_tasks(df: pd.DataFrame) -> list[tuple[str, str]]:
    texts = df["text"].tolist()
    texts_to_annotate = texts[:]
    return [(text, concept) for text in texts_to_annotate for concept in CONCEPTS]


def load_timing_results() -> pd.DataFrame:
    """Load timing results as DataFrame."""
    if not os.path.exists(TIMING_RESULTS_PATH):
        return pd.DataFrame()
    return pd.read_json(TIMING_RESULTS_PATH, lines=True)


def append_timing_result(result: dict):
    """Append a single timing result to the JSONL file (no deduplication)."""
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(TIMING_RESULTS_PATH), exist_ok=True)

    with open(TIMING_RESULTS_PATH, 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(f"Saved timing result: {result}")


def run_sweep(model: str, tasks: list[tuple[str, str]]) -> None:
    """Run annotation sweep for a single model across THINKING_OPTIONS.

    Stores timing information and saves the annotations produced by each model.
    """
    if is_local_model(model):
        _ = get_vllm_engine(model)  # Pre-load the model into GPU memory with vLLM

    for thinking in THINKING_OPTIONS:
        # Skip thinking=True for models without thinking support
        if thinking and not any(x in model for x in ("Qwen", "SmolLM3")):
            continue

        print(f"MODEL: {model}, THINKING: {thinking}")

        max_tokens = 512 if thinking else 3

        start_time = time.time()
        if 'gpt' in model:
            annotations = annotate(
                tasks,
                model=model,
                max_words_per_example=256,
                max_tokens=max_tokens,
                annotate_prompt_name="annotate-simple",
                max_retries=3,
            )
        else:
            annotations = annotate(
                tasks,
                model=model,
                max_words_per_example=256,
                max_tokens=max_tokens,
                annotate_prompt_name="annotate-simple",
                max_retries=3,
                tokenizer_kwargs={"enable_thinking": thinking},
            )
        duration = time.time() - start_time
        if duration > 10:
            timing_result = {
                'time (s)': round(duration, 1),
                'model_name': model,
                'thinking': thinking,
            }
            append_timing_result(timing_result)

        # ------------------------------------------------------------------
        # Store annotations
        # ------------------------------------------------------------------
        try:
            with open(ANNOTATIONS_PATH, 'rb') as f:
                all_annotations = pickle.load(f)
        except (FileNotFoundError, EOFError):
            all_annotations = {}

        model_key = f"{model}_think={thinking}"
        all_annotations[model_key] = annotations

        ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ANNOTATIONS_PATH, 'wb') as f:
            pickle.dump(all_annotations, f)


def print_timing_table():
    """Print timing results table."""
    df = load_timing_results()
    if df.empty:
        return
    
    print("\n" + "="*80)
    print("TIMING RESULTS:")
    print("="*80)
    print(df.to_csv(sep='\t', index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="HF model name to sweep. "
        "If omitted or set to 'all', the script runs once per model in MODELS.",
    )
    parser.add_argument("--data", default=str(DATA_PATH), help="JSONL file with Yelp reviews")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # OUTER LOOP: loop over models, start a subprocess for each one
    # -----------------------------------------------------------------------
    if args.model is None or args.model.lower() == "all":
        script_path = Path(__file__).resolve()
        for model_name in MODELS:
            print(f"\n=== Running sweep for {model_name} ===")
            try:
                subprocess.run(
                    [sys.executable, str(script_path), "--model", model_name, "--data", args.data],
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                print(f"Sweep failed for {model_name}: {exc}")
        
        print_timing_table()
        return

    # -----------------------------------------------------------------------
    # INNER LOOP: compute results for a single model
    # -----------------------------------------------------------------------
    df = pd.read_json(args.data, lines=True)
    tasks = build_tasks(df)
    run_sweep(args.model, tasks)


if __name__ == "__main__":
    main()