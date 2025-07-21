"""Local LLM utilities for HypotheSAEs."""
import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
torch.set_float32_matmul_precision("high")

from typing import List, Optional
from functools import lru_cache
from tqdm.auto import tqdm

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import HTTPError

from vllm import LLM, SamplingParams
import time

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
        print(f"Loading {model} in vLLM...")
        start_time = time.time()

        engine = LLM(model=model, task="generate")
        _LOCAL_ENGINES[model] = engine
        
        dtype = getattr(engine.llm_engine.get_model_config(), "dtype", "unknown")
        print(f"Loaded {model} with dtype: {dtype} (took {time.time() - start_time:.1f}s)")
    return engine

def get_local_completions(
    prompts: List[str],
    model: str = "Qwen/Qwen3-0.6B",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    show_progress: bool = True,
    tokenizer_kwargs: Optional[dict] = {},
    sampling_kwargs: Optional[dict] = {},
) -> List[str]:
    """Generate completions using vLLM with llm.generate()."""
    engine = _get_engine(model)
    tokenizer = engine.get_tokenizer()

    if getattr(tokenizer, "chat_template", None) is not None:
        messages_lists = [[{"role": "user", "content": p}] for p in prompts]
        prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **tokenizer_kwargs)
                   for messages in messages_lists]

    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, **sampling_kwargs)
    outputs = engine.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=show_progress,
    )

    completions = [str(out.outputs[0].text) for out in outputs]
    return completions