"""Local LLM utilities for HypotheSAEs."""

from typing import List, Optional
import torch
torch.set_float32_matmul_precision("high")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset

from tqdm.auto import tqdm

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
        print(f"Loaded {model}, using {pipe.device} and precision {pipe.model.dtype}")
    return pipe


def get_local_completions(
    prompts: List[str],
    model: str = "google/gemma-2-2b-it",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate completions for a list of prompts using a local model.
    
    Args:
        prompts: Prompts to complete.
        model: HF model ID or local name registered with `_get_pipeline`.
        batch_size: Batch size given to the HF pipeline.
        max_new_tokens: Max tokens to generate per prompt.
        temperature: Sampling temperature (0 ⇒ greedy).
        show_progress: If True, wrap generation in a tqdm progress bar.
        progress_desc: Label shown in the progress bar, if enabled.
    """
    pipe = _get_pipeline(model)
    ds   = Dataset.from_dict({"text": prompts})
    
    generator = pipe(
        KeyDataset(ds, "text"),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )

    if show_progress:
        generator = tqdm(generator, total=len(prompts), desc=progress_desc)

    return [item[0]["generated_text"] for item in generator]

