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

_LOCAL_ENGINES: dict[str, LLM] = {}

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

def _sleep_all_except(active_model: Optional[str] = None) -> None:
    """Put every cached vLLM engine *except* `active` to sleep."""
    for name, engine in _LOCAL_ENGINES.items():
        if name == active_model:
            continue
        if engine.llm_engine.is_sleeping():
            continue
        print(f"Sleeping {name} to free GPU memory...")
        engine.llm_engine.reset_prefix_cache()
        engine.sleep(level=2) # Level 1 clears KV cache and moves weights to CPU; Level 2 clears cache + clears weights entirely

def get_vllm_engine(model: str, **kwargs) -> LLM:
    """
    Return a vLLM engine for `model`.

    * If the engine is already cached, sleep the others and wake it.
    * If it is not cached, sleep every other engine first so the GPU
      is empty, then load the new model.
    """
    engine = _LOCAL_ENGINES.get(model)

    if engine is None:
        _sleep_all_except(active_model=None) # free GPU before allocating

        print(f"Loading {model} in vLLM...")
        t0 = time.time()
        gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", 0.85)
        engine = LLM(model=model, task="generate", enable_sleep_mode=True, gpu_memory_utilization=gpu_memory_utilization, **kwargs)
        _LOCAL_ENGINES[model] = engine
        dtype = getattr(engine.llm_engine.get_model_config(), "dtype", "unknown")
        print(f"Loaded {model} with dtype: {dtype} (took {time.time()-t0:.1f}s)")
    else:
        _sleep_all_except(active_model=model)
        if engine.llm_engine.is_sleeping(): 
            print(f"Engine found for {model} but model is sleeping, waking up...")
            print(f"[WARNING]: This functionality is currently bugged in vLLM, where you may see nonsense outputs after waking up a model.")
            engine.wake_up()
            engine.llm_engine.reset_prefix_cache()

    return engine

def get_local_completions(
    prompts: List[str],
    model: str = "Qwen/Qwen3-0.6B",
    max_tokens: int = 128,
    show_progress: bool = True,
    tokenizer_kwargs: Optional[dict] = {},
    llm_sampling_kwargs: Optional[dict] = {},
) -> List[str]:
    """Generate completions using vLLM with llm.generate()."""
    engine = get_vllm_engine(model)
    tokenizer = engine.get_tokenizer()

    if getattr(tokenizer, "chat_template", None) is not None:
        messages_lists = [[{"role": "user", "content": p}] for p in prompts]
        enable_thinking = tokenizer_kwargs.pop("enable_thinking", False) # Default to False so users don't get unexpected output
        prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking, **tokenizer_kwargs)
                   for messages in messages_lists]

    sampling_params = SamplingParams(max_tokens=max_tokens, **llm_sampling_kwargs)
    outputs = engine.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=show_progress,
    )

    completions = [str(out.outputs[0].text) for out in outputs]
    return completions