"""Local LLM utilities for HypotheSAEs."""

from typing import List, Optional, Iterator
import torch
torch.set_float32_matmul_precision("high")

from functools import lru_cache

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import HTTPError

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from tqdm.auto import tqdm

_LOCAL_PIPES = {}

@lru_cache(maxsize=256) # Cache models that we've already checked
def hf_model_exists(repo_id: str) -> bool:
    try:
        HfApi().model_info(repo_id, timeout=3)
        return True                              
    except RepositoryNotFoundError:
        return False
    except HTTPError as e:
        return e.response is not None and e.response.status_code in {401, 403}

def is_local_model(model: str) -> bool:
    return model in _LOCAL_PIPES or hf_model_exists(model)

def _get_pipeline(model: str):
    """Load and cache a text generation pipeline for the given model."""
    pipe = _LOCAL_PIPES.get(model)
    if pipe is None:
        if 'qwen3' in model.lower():
            print("WARNING: Currently, Qwen3 models and other `thinking` models may not have their outputs parsed correctly.")
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
            return_full_text=False, # if True, returns full message history; if False, returns only the new generated text
        )

        if pipe.tokenizer.pad_token is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        if pipe.tokenizer.padding_side == "right":
            pipe.tokenizer.padding_side = "left"

        _LOCAL_PIPES[model] = pipe
        print(f"Loaded {model} (device: {pipe.device}; dtype: {pipe.model.dtype})")
    return pipe


def get_local_completions(
    prompts: List[str],
    model: str = "google/gemma-3-1b-it",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate completions for *prompts* with real‑time progress feedback."""
    pipe = _get_pipeline(model)
    is_chat_model = getattr(pipe.tokenizer, "chat_template", None) is not None

    # If chat model, stream prompts in openai 'messages' format; otherwise use raw strings
    prompt_iter: Iterator = (
        ( [{"role": "user", "content": p}] if is_chat_model else p ) for p in prompts
    )

    stream = pipe(
        prompt_iter,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature or None,
    )

    completions: List[str] = []
    pbar = tqdm(total=len(prompts), desc=progress_desc, disable=not show_progress)

    for batch in stream:
        for out in batch:
            txt = out["generated_text"]
            # if output is in chat messages format, extract final response (assistant)
            if isinstance(txt, list) and txt and isinstance(txt[-1], dict):
                txt = txt[-1].get("content", "")
            completions.append(str(txt))
        pbar.update(len(batch))

    pbar.close()
    return completions
