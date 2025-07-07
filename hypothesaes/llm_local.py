"""Local LLM utilities for HypotheSAEs."""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

_LOCAL_PIPES = {}

def is_local_model(model: str) -> bool:
    """Return True if model should be treated as a local HF model."""
    return "gemma-2" in model or "/" in model or model in _LOCAL_PIPES


def _get_pipeline(model: str):
    """Load and cache a text generation pipeline for the given model."""
    pipe = _LOCAL_PIPES.get(model)
    if pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model_obj,
            tokenizer=tokenizer,
            return_full_text=False,
        )
        if pipe.tokenizer.pad_token is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        _LOCAL_PIPES[model] = pipe
    return pipe


def get_local_completions(
    prompts: List[str],
    model: str = "google/gemma-2-2b-it",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> List[str]:
    """Generate completions for a list of prompts using a local model."""
    pipe = _get_pipeline(model)
    completions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outputs = pipe(
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            batch_size=len(batch),
        )
        completions.extend([out[0]["generated_text"] for out in outputs])
    return completions
