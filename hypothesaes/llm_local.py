"""Local LLM utilities for HypotheSAEs."""

from typing import List, Optional
import torch
torch.set_float32_matmul_precision("high")

from functools import lru_cache

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import HTTPError

from vllm import LLM, SamplingParams

from tqdm.auto import tqdm

_LOCAL_ENGINES = {}

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
    return model in _LOCAL_ENGINES or hf_model_exists(model)

def _get_engine(model: str) -> LLM:
    """Load and cache a vLLM engine for the given model."""
    engine = _LOCAL_ENGINES.get(model)
    if engine is None:
        if 'qwen3' in model.lower():
            print("WARNING: Currently, Qwen3 models and other `thinking` models may not have their outputs parsed correctly.")
        engine = LLM(model=model, dtype="float32")
        tokenizer = engine.get_tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.padding_side == "right":
            tokenizer.padding_side = "left"
        _LOCAL_ENGINES[model] = engine
        dtype = getattr(engine.llm_engine.get_model_config(), "dtype", "unknown")
        print(f"Loaded {model} (dtype: {dtype})")
    return engine


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
    engine = _get_engine(model)
    tokenizer = engine.get_tokenizer()
    is_chat_model = getattr(tokenizer, "chat_template", None) is not None

    if is_chat_model:
        messages = [[{"role": "user", "content": p}] for p in prompts]
        outputs = engine.chat(
            messages,
            sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=temperature),
            use_tqdm=show_progress,
        )
    else:
        outputs = engine.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=temperature),
            use_tqdm=show_progress,
        )

    completions: List[str] = []
    for out in outputs:
        completions.append(str(out.outputs[0].text))

    return completions
