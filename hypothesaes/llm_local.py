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
        print(f"Loading {model} in vLLM...")
        engine = LLM(model=model, task="generate")
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
    model: str = "google/gemma-3-12b-it",
    max_batch_tokens: int = 4096,  # Adjust based on model size
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate completions with token-based batching."""
    engine = _get_engine(model)
    tokenizer = engine.get_tokenizer()
    is_chat_model = getattr(tokenizer, "chat_template", None) is not None
    
    # Pre-tokenize to count tokens
    if is_chat_model:
        formatted_prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], 
                                        tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
    else:
        formatted_prompts = prompts
    
    # Tokenize all prompts to get lengths
    tokenized = tokenizer(formatted_prompts, add_special_tokens=True)
    token_lengths = [len(ids) for ids in tokenized['input_ids']]
    
    # Create batches based on token count
    batches = []
    current_batch = []
    current_batch_indices = []
    current_token_count = 0
    
    for i, (prompt, token_len) in enumerate(zip(prompts, token_lengths)):
        # Account for both input and max output tokens
        total_tokens_for_prompt = token_len + max_new_tokens
        
        if current_token_count + total_tokens_for_prompt > max_batch_tokens and current_batch:
            batches.append((current_batch, current_batch_indices))
            current_batch = []
            current_batch_indices = []
            current_token_count = 0
        
        current_batch.append(prompt)
        current_batch_indices.append(i)
        current_token_count += total_tokens_for_prompt
    
    if current_batch:
        batches.append((current_batch, current_batch_indices))
    
    # Process batches
    all_completions = [None] * len(prompts)
    
    for batch_prompts, batch_indices in tqdm(batches, 
                                            desc=progress_desc or f"Processing {len(batches)} batches",
                                            disable=not show_progress):
        if is_chat_model:
            messages = [[{"role": "user", "content": p}] for p in batch_prompts]
            outputs = engine.chat(
                messages,
                sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=temperature),
                use_tqdm=False,
            )
        else:
            outputs = engine.generate(
                batch_prompts,
                sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=temperature),
                use_tqdm=False,
            )
        
        # Place results in correct positions
        for idx, output in zip(batch_indices, outputs):
            all_completions[idx] = str(output.outputs[0].text)
    print(outputs)
    print(all_completions)
    return all_completions

# def get_local_completions(
#     prompts: List[str],
#     model: str = "google/gemma-3-1b-it",
#     batch_size: int = 4,
#     max_new_tokens: int = 128,
#     temperature: float = 0.7,
#     show_progress: bool = True,
#     progress_desc: Optional[str] = None,
# ) -> List[str]:
#     """Generate completions for *prompts* with real‑time progress feedback."""
#     engine = _get_engine(model)
#     tokenizer = engine.get_tokenizer()
#     is_chat_model = getattr(tokenizer, "chat_template", None) is not None

#     if is_chat_model:
#         messages = [[{"role": "user", "content": p}] for p in prompts]
#         print("Generating completions for chat model...")
#         outputs = engine.chat(
#             messages,
#             sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=temperature),
#             use_tqdm=show_progress,
#         )
#     else:
#         print("Generating completions for non-chat model...")
#         outputs = engine.generate(
#             prompts,
#             sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=temperature),
#             use_tqdm=show_progress,
#         )

#     completions = [str(out.outputs[0].text) for out in outputs]
#     print(completions)
#     return completions