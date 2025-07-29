#!/usr/bin/env python3
"""Generate three SAE-neuron interpretations per model/setting and record runtime."""

import argparse
import sys, json, time, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from hypothesaes.llm_local import get_vllm_engine, is_local_model
from hypothesaes.interpret_neurons import NeuronInterpreter, InterpretConfig, LLMConfig, SamplingConfig, sample_percentile_bins

# Defaults ------------------------------------------------------------------
M, K = 1024, 32

ROOT = Path('/nas/ucb/rmovva/data/hypothesaes')
DATA_PATH = ROOT / "hypothesis-generation-data/yelp/train-200K.json"
ACTIVATIONS_PATH = ROOT / f"local_llm_experiments/checkpoints/activations_{M}_{K}.npy"
NEURON_SELECTION_SEED = 42
N_NEURONS_TO_INTERPRET = 256
N_CANDIDATE_INTERPRETATIONS = 3

TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are reviews of restaurants on Yelp.
Features should describe a specific aspect of the review. For example:
- "mentions long wait times to receive service"
- "praises how a dish was cooked, with phrases like 'perfect medium-rare'\""""

THINKING_OPTIONS = [True, False]

# ---------------------------------------------------------------------------
# List of models to sweep when no --model argument is provided
# ---------------------------------------------------------------------------
MODELS = [
    "Qwen/Qwen3-8B-AWQ",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B-AWQ",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B-AWQ",
    "Qwen/Qwen3-32B",
    #
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
]
TIMING_RESULTS_PATH = ROOT / f"local_llm_experiments/results/interpretation_timing_results_{M}_{K}.jsonl"
INTERPRETATIONS_PATH = ROOT / f"local_llm_experiments/results/interpretations_{M}_{K}.json"


def append_timing(result: dict) -> None:
    """Append timing result as a JSON-line."""
    TIMING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMING_RESULTS_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")


def run_sweep(model: str, texts: list[str], activations: np.ndarray, neurons_to_interpret: list[int], n_candidate_interpretations: int) -> None:
    """Generate 3 candidate interpretations per neuron using the specified model."""
    if is_local_model(model):
        _ = get_vllm_engine(model)  # Pre-load the model into GPU memory with vLLM

    interpreter = NeuronInterpreter(interpreter_model=model)
    for thinking in THINKING_OPTIONS:
        # Skip thinking=True for models without thinking support
        if thinking and not any(x in model for x in ("Qwen", "SmolLM3")):
            continue

        print(f"MODEL: {model}, THINKING: {thinking}")
        model_key = f"{model}_think={thinking}"

        max_tokens = 8192 if thinking else 50
        config = InterpretConfig(
            sampling=SamplingConfig(
                function=sample_percentile_bins,
                n_examples=10,
                sampling_kwargs={"high_percentile": (80, 100), "low_percentile": None},
            ),
            llm=LLMConfig(
                max_interpretation_tokens=max_tokens,
                tokenizer_kwargs={"enable_thinking": thinking},
            ),
            prompt_name="interpret-neuron-binary-reasoning" if thinking else "interpret-neuron-binary",
            n_candidates=n_candidate_interpretations,  # generate three interpretations per neuron
            task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS,
        )

        start_time = time.time()
        interpretations = interpreter.interpret_neurons(
            texts=texts,
            activations=activations,
            neuron_indices=neurons_to_interpret,
            config=config,
        )
        duration = time.time() - start_time

        timing_result = {
            'time (s)': round(duration, 1),
            'model_name': model,
            'thinking': thinking,
        }
        append_timing(timing_result)

        try:
            with open(INTERPRETATIONS_PATH, "r") as f:
                all_interps = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_interps = {}

        # JSON requires str keys
        model_dict = {str(n): v for n, v in interpretations.items()}
        all_interps[model_key] = model_dict

        with open(INTERPRETATIONS_PATH, "w") as f:
            json.dump(all_interps, f, indent=2)


def print_timing_table():
    """Print timing results table."""
    df = pd.read_json(TIMING_RESULTS_PATH, lines=True)
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
    parser.add_argument("--activations", default=str(ACTIVATIONS_PATH), help="NumPy file containing SAE activations")
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
                    [sys.executable, str(script_path), "--model", model_name, "--data", args.data, "--activations", args.activations],
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
    texts = df["text"].tolist()
    activations = np.load(args.activations)

    neurons_to_interpret = np.random.RandomState(NEURON_SELECTION_SEED).choice(activations.shape[1], size=N_NEURONS_TO_INTERPRET, replace=False).tolist()
    if activations.shape[0] != len(texts):
        raise ValueError(f"Number of activations ({activations.shape[0]}) does not match number of texts ({len(texts)})")

    run_sweep(args.model, texts, activations, neurons_to_interpret, N_CANDIDATE_INTERPRETATIONS)


if __name__ == "__main__":
    main()