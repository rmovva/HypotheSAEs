"""Text annotation using LLM-based concept checking."""

import numpy as np
from typing import List, Optional, Dict, Tuple
import concurrent.futures
from tqdm.auto import tqdm
import os
import json
from pathlib import Path
import time

from .llm_api import get_completion
from .utils import load_prompt, truncate_text

CACHE_DIR = os.path.join(Path(__file__).parent.parent, 'annotation_cache')
DEFAULT_N_WORKERS = 30 

def get_annotation_cache(cache_path: str) -> dict:
    """Load cached annotations from JSON file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse cache file {cache_path}, starting fresh cache")
            os.remove(cache_path)
    return {}

def save_annotation_cache(cache_path: str, cache: dict) -> None:
    """Save annotations to JSON cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f)

def generate_cache_key(concept: str, text: str) -> str:
    """Generate a cache key for a given concept and text."""
    return f"{concept}|||{text[:100]}...{text[-100:]}"

def annotate_single_text(
    text: str,
    concept: str,
    model: str = "gpt-4o-mini",
    max_words_per_example: Optional[int] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: float = 1.0
) -> Tuple[Optional[int], float]:  # Return tuple of (result, api_time)
    """
    Annotate a single text with given concept using LLM.
    Returns (annotation, api_time) where annotation is 1 (present), 0 (absent), or None (failed).
    """
    if max_words_per_example:
        text = truncate_text(text, max_words_per_example)
        
    prompt_template = load_prompt("annotate")
    prompt = prompt_template.format(hypothesis=concept, text=text)
    
    total_api_time = 0.0
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response_text = get_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=1,
                timeout=timeout
            ).strip().lower()
            total_api_time += time.time() - start_time
            
            if response_text == "yes":
                return 1, total_api_time
            elif response_text == "no":
                return 0, total_api_time
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to annotate after {max_retries} attempts: {e}")
                return None, total_api_time
            continue
    
    return None, total_api_time

def _parallel_annotate(
    tasks: List[Tuple[str, str]],
    n_workers: int,
    cache: dict,
    cache_path: Optional[str],
    results: Dict[str, Dict[str, int]],
    progress_desc: str = "Annotating",
    show_progress: bool = True,
    **kwargs
) -> None:
    # Keep track of tasks that need to be retried
    retry_tasks = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {
            executor.submit(annotate_single_text, text=text, concept=concept, **kwargs): 
            (text, concept)
            for text, concept in tasks
        }
        
        iterator = tqdm(concurrent.futures.as_completed(future_to_task), 
                       total=len(tasks),
                       desc=progress_desc,
                       disable=not show_progress)
        
        for future in iterator:
            text, concept = future_to_task[future]
            try:
                annotation, _ = future.result()
                if annotation is not None:
                    if concept not in results:
                        results[concept] = {}
                    results[concept][text] = annotation
                    if cache_path:
                        cache[generate_cache_key(concept, text)] = annotation
            except Exception as e:
                retry_tasks.append((text, concept))
                print(f"Failed to annotate text for concept '{concept}': {e}")
    
    # Retry failed tasks sequentially
    if retry_tasks:
        print(f"Retrying {len(retry_tasks)} failed tasks...")
        for text, concept in retry_tasks:
            try:
                annotation, _ = annotate_single_text(text=text, concept=concept, **kwargs)
                if annotation is not None:
                    if concept not in results:
                        results[concept] = {}
                    results[concept][text] = annotation
                    if cache_path:
                        cache[generate_cache_key(concept, text)] = annotation
            except Exception as e:
                print(f"Failed to annotate text for concept '{concept}' during retry: {e}")

def annotate(
    tasks: List[Tuple[str, str]],
    cache_path: Optional[str] = None,
    n_workers: int = DEFAULT_N_WORKERS,
    show_progress: bool = True,
    **kwargs
) -> Dict[Tuple[str, str], int]:
    """
    Annotate a list of (text, concept) tasks.
    
    Args:
        tasks: List of (text, concept) tuples to annotate
        cache_path: Path to cache file
        n_workers: Number of workers for parallel processing
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments passed to annotate_single_text
    
    Returns:
        Dictionary mapping (text, concept) to annotation result
    """
    # Load existing cache
    cache = get_annotation_cache(cache_path) if cache_path else {}
    results = {}
    uncached_tasks = []

    # Check cache and prepare uncached tasks
    for text, concept in tasks:
        if concept not in results:
            results[concept] = {}
        cache_key = generate_cache_key(concept, text)
        if cache_key in cache:
            results[concept][text] = cache[cache_key]
        else:
            uncached_tasks.append((text, concept))

    # Print cache statistics
    print(f"Found {len(tasks) - len(uncached_tasks)} cached items; annotating {len(uncached_tasks)} uncached items")

    # Annotate uncached tasks
    if uncached_tasks:
        _parallel_annotate(
            tasks=uncached_tasks,
            n_workers=n_workers,
            cache=cache,
            cache_path=cache_path,
            results=results,
            show_progress=show_progress,
            **kwargs
        )

    # Save cache if path provided
    if cache_path:
        save_annotation_cache(cache_path, cache)

    return results

def annotate_texts_with_concepts(
    texts: List[str],
    concepts: List[str],
    cache_name: Optional[str] = None,
    progress_desc: str = "Annotating",
    show_progress: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Annotate all texts in a list with all concepts in a list.
    Returns:
        Dictionary mapping each concept to an array of annotation results, with the texts in the order they were passed in.
    """
    # Create tasks for each text-concept pair
    tasks = [(text, concept) for text in texts for concept in concepts]
    
    # Use the annotate function to process tasks
    results = annotate(
        tasks=tasks,
        cache_path=os.path.join(CACHE_DIR, f"{cache_name}_hypothesis-eval.json") if cache_name else None,
        n_workers=kwargs.pop('n_workers', DEFAULT_N_WORKERS),
        show_progress=show_progress,
        progress_desc=progress_desc,
        **kwargs
    )
    
    # Reorganize results into arrays per concept
    concept_arrays = {}
    for concept in concepts:
        concept_arrays[concept] = np.array([results[concept][text] for text in texts])
        
    return concept_arrays
