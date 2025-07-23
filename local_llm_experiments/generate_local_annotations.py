#!/usr/bin/env python3
"""
Sweep annotate cache builder.

Example
-------
python sweep_annotate.py --model 'Qwen/Qwen3-0.6B'
"""

import argparse
import os
import time
import json

import numpy as np
import pandas as pd
from hypothesaes.annotate import annotate
from hypothesaes.annotate import CACHE_DIR
from hypothesaes.llm_local import _get_engine

from pathlib import Path
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration – edit as needed
# ---------------------------------------------------------------------------

current_dir = os.getcwd()
prefix = './' if current_dir.endswith("HypotheSAEs") else '../'
DATA_PATH = os.path.join(prefix, "demo-data", "yelp-demo-val-2K.json")

CONCEPTS = [
    "uses superlative language to describe the restaurant as the best in a specific category (e.g., 'best lobster roll', 'best bakery', 'best cheesesteaks')",
    "mentions experiences of food causing illness, such as food poisoning, stomach pain, or vomiting",
    "mentions long wait times for seating, food, or service",
    "complains about repeated service errors or unresolved issues despite multiple attempts to address them",
    "complains about poor or rude customer service",
]

TEMPERATURES = [0.0]
THINKING_OPTIONS = [True, False]
PROMPT_TEMPLATES = ["annotate-simple", "annotate"]

# ---------------------------------------------------------------------------
# List of models to sweep when no --model argument is provided
# ---------------------------------------------------------------------------
MODELS = [
    "Qwen/Qwen3-0.6B",
    "google/gemma-3-1b-it",
    "HuggingFaceTB/SmolLM3-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B",
    "google/gemma-3-4b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3-14B",
]
TIMING_RESULTS_PATH = os.path.join('.', 'results', "timing_results_annotation.jsonl")


def build_tasks(df: pd.DataFrame) -> list[tuple[str, str]]:
    texts = df["text"].tolist()
    texts_to_annotate = texts[:]
    return [(text, concept) for text in texts_to_annotate for concept in CONCEPTS]


def load_timing_results() -> pd.DataFrame:
    """Load timing results as DataFrame."""
    if not os.path.exists(TIMING_RESULTS_PATH):
        return pd.DataFrame()
    
    try:
        return pd.read_json(TIMING_RESULTS_PATH, lines=True)
    except:
        return pd.DataFrame()


def save_timing_result(result: dict):
    """Save a single timing result to JSONL file."""
    df_existing = load_timing_results()
    
    # Check if result already exists (same values except for "time (s)")
    result_without_time = {k: v for k, v in result.items() if k != 'time (s)'}
    
    if not df_existing.empty:
        df_without_time = df_existing.drop(columns=['time (s)'], errors='ignore')
        for _, row in df_without_time.iterrows():
            if row.to_dict() == result_without_time:
                print(f"Timing result already exists, skipping: {result}")
                return
    
    # Append new result
    with open(TIMING_RESULTS_PATH, 'a') as f:
        f.write(json.dumps(result) + '\n')
    
    print(f"Saved timing result: {result}")


def run_sweep(model: str, tasks: list[tuple[str, str]]) -> None:
    _ = _get_engine(model) # Pre-load the model into GPU memory with vLLM
    for temperature in TEMPERATURES:
        for thinking in THINKING_OPTIONS:
            for prompt_template in PROMPT_TEMPLATES:
                # Skip combinations that don't support "thinking", mirroring notebook logic
                if thinking and not any(x in model for x in ("Qwen", "SmolLM3")):
                    continue
                
                print(f"MODEL: {model}, TEMPERATURE: {temperature}, THINKING: {thinking}, PROMPT: {prompt_template}")
                max_tokens = 512 if thinking else 3
                cache_path = os.path.join(CACHE_DIR, f'test_{model}_temp={temperature}_think={thinking}_prompt={prompt_template}.json')

                start_time = time.time()
                annotate(
                    tasks,
                    model=model,
                    max_words_per_example=256,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    prompt_template_name=prompt_template,
                    tokenizer_kwargs={"enable_thinking": thinking},
                    max_retries=3,
                    cache_path=cache_path,
                )
                end_time = time.time()
                duration = end_time - start_time
                
                # Only store timing if >10 seconds (avoid cached results)
                if duration > 10:
                    timing_result = {
                        'time (s)': duration,
                        'model_name': model,
                        'temperature': temperature,
                        'prompt': prompt_template,
                        'thinking': thinking,
                    }
                    save_timing_result(timing_result)


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
    # OUTER LOOP: if no specific model was given, spawn a subprocess per model
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
                print(f"⚠️  Sweep failed for {model_name}: {exc}")
        
        print_timing_table()
        return

    # -----------------------------------------------------------------------
    # INNER LOOP: normal behaviour when --model is provided
    # -----------------------------------------------------------------------
    df = pd.read_json(args.data, lines=True)
    tasks = build_tasks(df)
    run_sweep(args.model, tasks)


if __name__ == "__main__":
    main()