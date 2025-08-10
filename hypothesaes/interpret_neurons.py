"""Methods for interpreting SAE neurons using LLMs."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Any
from tqdm.auto import tqdm
import concurrent.futures
import os
from dataclasses import dataclass, field

from .llm_api import get_completion
from .llm_local import is_local_model, get_local_completions
from .utils import load_prompt, truncate_text
from .annotate import annotate, CACHE_DIR

DEFAULT_TASK_SPECIFIC_INSTRUCTIONS = """An example feature could be:
- "uses multiple adjectives to describe colors"
- "describes a patient experiencing seizures or epilepsy"
- "contains multiple single-digit numbers\""""

DEFAULT_MAX_INTERPRETATION_TOKENS_THINKING = 8192

def sample_top_zero(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    max_words_per_example: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """Sample top activating examples and random zero-activation examples for a given neuron."""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    neuron_acts = activations[:, neuron_idx]
    n_per_class = n_examples // 2
    
    # Get indices of positive activations and take top n_per_class (or fewer if not enough positive)
    count_positive_activating = np.sum(neuron_acts > 0)
    if count_positive_activating < n_per_class:
        print(f"[WARNING] Only found {count_positive_activating} examples with positive activation, using all available")
        top_indices = np.argsort(neuron_acts)[-count_positive_activating:]
    else:
        top_indices = np.argsort(neuron_acts)[-n_per_class:]
    
    # Get zero activation examples
    zero_indices = np.where(neuron_acts == 0)[0]
    if len(zero_indices) >= n_per_class:
        random_indices = np.random.choice(zero_indices, size=n_per_class, replace=False)
    else:
        print(f"[WARNING] Only found {len(zero_indices)} examples with zero activation, using all available")
        random_indices = zero_indices
    
    pos_texts = [texts[i] for i in top_indices]
    neg_texts = [texts[i] for i in random_indices]
    
    if max_words_per_example:
        pos_texts = [truncate_text(text, max_words_per_example) for text in pos_texts]
        neg_texts = [truncate_text(text, max_words_per_example) for text in neg_texts]
    
    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "positive_activations": neuron_acts[top_indices].tolist(),
        "negative_activations": neuron_acts[random_indices].tolist()
    }

def sample_percentile_bins(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    max_words_per_example: Optional[int] = None,
    high_percentile: Tuple[float, float] = (90, 100),
    low_percentile: Optional[Tuple[float, float]] = None,
    random_seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """Sample examples from high activation percentile bins and either low percentile or zero activations."""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    neuron_acts = activations[:, neuron_idx]
    n_per_class = n_examples // 2
    
    pos_mask = neuron_acts > 0
    pos_vals = neuron_acts[pos_mask]
    pos_indices = np.where(pos_mask)[0]
    
    high_mask = (pos_vals >= np.percentile(pos_vals, high_percentile[0])) & \
               (pos_vals <= np.percentile(pos_vals, high_percentile[1]))
    high_indices = pos_indices[high_mask]
    if len(high_indices) >= n_per_class:
        high_sample_indices = np.random.choice(high_indices, size=n_per_class, replace=False)
    else:
        print(f"[WARNING] There are less than {n_per_class} examples in bin {high_percentile} for neuron {neuron_idx}; using {len(high_indices)} instead")
        high_sample_indices = high_indices
    
    if low_percentile is not None:
        low_mask = (pos_vals >= np.percentile(pos_vals, low_percentile[0])) & \
                  (pos_vals <= np.percentile(pos_vals, low_percentile[1]))
        low_indices = pos_indices[low_mask]
    else: # Use examples with zero activation as the negative examples
        low_indices = np.where(neuron_acts == 0)[0]

    if len(low_indices) >= n_per_class:
        low_sample_indices = np.random.choice(low_indices, size=n_per_class, replace=False)
    else:
        print(f"[WARNING] There are less than {n_per_class} examples in bin {low_percentile} for neuron {neuron_idx}; using {len(low_indices)} instead")
        low_sample_indices = low_indices
    
    pos_texts = [texts[i] for i in high_sample_indices]
    neg_texts = [texts[i] for i in low_sample_indices]
    
    if max_words_per_example:
        pos_texts = [truncate_text(text, max_words_per_example) for text in pos_texts]
        neg_texts = [truncate_text(text, max_words_per_example) for text in neg_texts]
    
    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "positive_activations": neuron_acts[high_sample_indices].tolist(),
        "negative_activations": neuron_acts[low_sample_indices].tolist()
    }

def sample_custom(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    random_seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """Sample examples using a custom function.
    
    This function should return a dictionary with keys that correspond to your prompt template.
    The default prompt template is "interpret-neuron-binary.txt", which expects two keys: positive_samples and negative_samples.

    For example, you can write a custom sampling function that outputs:
    - only top-activating examples
    - top-activating, medium-activating, and non-activating examples
    - etc.

    Note that if you change the sampling setup, you will also need to write a new prompt template.
    Ensure that the keys in your output dictionary match the keys in your prompt template.
    
    Args:
        texts: List of all text examples
        activations: Neuron activation matrix (n_samples, n_neurons)
        neuron_idx: Index of neuron to sample examples for
        [any other arguments]
    """
    pass

@dataclass
class SamplingConfig:
    function: Callable = sample_top_zero
    n_examples: int = 20 # Number of examples to sample to prompt the interpreter
    random_seed: Optional[int] = 0 # Base random seed for example sampling; each interp candidate increments this seed by 1
    max_words_per_example: Optional[int] = 256 # Maximum number of words per text example, truncated if necessary
    sampling_kwargs: Dict[str, Any] = field(default_factory=dict) # Extra keyword arguments for the sampling function

@dataclass
class LLMConfig:
    temperature: Optional[float] = None # Temperature for the interpreter model
    max_interpretation_tokens: int = 100 # Maximum number of tokens for each generated interpretation
    timeout: float = 10.0 # Timeout for the interpreter model (in seconds)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {"enable_thinking": True}) # tokenizer_kwargs are not used for API models

@dataclass
class InterpretConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    n_candidates: int = 1 # Number of candidate interpretations per neuron
    interpretation_prompt_name: str = "interpret-neuron-binary" # Name of the prompt template file to use
    task_specific_instructions: str = DEFAULT_TASK_SPECIFIC_INSTRUCTIONS # Task-specific instructions for the interpreter model

@dataclass
class ScoringConfig:
    n_examples: int = 100 # Number of examples to score interpretation fidelity (half top-activating, half zero-activating)
    max_words_per_example: Optional[int] = 256 # Maximum number of words per text example, truncated if necessary
    sampling_function: Callable = sample_top_zero # Function to sample examples for scoring
    sampling_kwargs: Dict[str, Any] = field(default_factory=dict) # Extra keyword arguments for the sampling function

class NeuronInterpreter:
    def __init__(
        self,
        interpreter_model: str = "gpt-4.1",
        annotator_model: str = "gpt-4.1-mini",
        n_workers_interpretation: int = 10,
        n_workers_annotation: int = 30,
        cache_name: Optional[str] = None,
    ):
        """Initialize a NeuronInterpreter."""
        self.interpreter_model = interpreter_model
        self.annotator_model = annotator_model
        self.n_workers_interpretation = n_workers_interpretation
        self.n_workers_annotation = n_workers_annotation
        self.cache_name = cache_name

    def _build_interpretation_prompt(
        self,
        texts: List[str],
        activations: np.ndarray,
        neuron_idx: int,
        candidate_idx: int, # This is just used to ensure different sampling seeds for each candidate
        config: InterpretConfig,
    ) -> Optional[str]:
        """Return a fully-formatted prompt for a given neuron or ``None`` if the neuron is dead."""
        if np.all(activations[:, neuron_idx] <= 0):
            print(f"[WARNING] All activations for neuron {neuron_idx} are <= 0. This neuron may be dead. Skipping interpretation.")
            return None

        formatted_examples = config.sampling.function(
            texts=texts,
            activations=activations,
            neuron_idx=neuron_idx,
            n_examples=config.sampling.n_examples,
            max_words_per_example=config.sampling.max_words_per_example,
            random_seed=config.sampling.random_seed + candidate_idx,
            **config.sampling.sampling_kwargs,
        )

        try:
            interpretation_prompt_template = load_prompt(config.interpretation_prompt_name)
            prompt = interpretation_prompt_template.format(
                task_specific_instructions=config.task_specific_instructions,
                **formatted_examples
            )
        except KeyError as e:
            raise KeyError(f"Missing required key {e} in the interpretation prompt template. Please ensure all required keys are provided in formatted_examples.")

        return prompt
    
    def _parse_interpretation(self, response: str) -> str:
        """Parse raw LLM response into clean interpretation string."""
        response = response.strip()
        
        # Handle incomplete response (i.e. if the model started thinking but didn't finish)
        if '<think>' in response and '</think>' not in response:
            return None
        # Thinking completed
        if '</think>' in response:
            response = response.split('</think>')[1].strip()
        
        # Remove any prefixes
        response = response.split('\n', 1)[0]
        prefixes = ['- ', '"-', '" -']
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):]
        
        return response.strip('"').strip()
 
    def _get_interpretation_openai(
        self,
        prompt: str,
        config: InterpretConfig,
    ) -> Optional[str]:
        """Send a single prompt to the interpreter model and return the parsed interpretation."""
        try:
            if "gpt-5" in self.interpreter_model:
                response = get_completion(
                    prompt=prompt,
                    model=self.interpreter_model,
                    max_completion_tokens=config.llm.max_interpretation_tokens,
                    reasoning_effort="minimal",
                )
            elif self.interpreter_model.startswith("o"):
                response = get_completion(
                    prompt=prompt,
                    model=self.interpreter_model,
                    max_completion_tokens=config.llm.max_interpretation_tokens,
                    reasoning_effort="low",
                )
            else:
                response = get_completion(
                    prompt=prompt,
                    model=self.interpreter_model,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_interpretation_tokens,
                    timeout=config.llm.timeout,
                )
            return self._parse_interpretation(response)
        except Exception as e:
            print(f"Failed to get interpretation: {e}")
            return None

    def _execute_prompts(
        self,
        prompts: List[str],
        config: InterpretConfig,
    ) -> List[str]:
        """Execute a batch of prompts and return parsed interpretations (one per prompt)."""
        # Remove ``None`` before sending for completion
        valid_prompts = [p for p in prompts if p is not None]
        if not valid_prompts:
            return []

        # For local models (vLLM), use a single call to get_local_completions
        llm_sampling_kwargs = {"temperature": config.llm.temperature} if config.llm.temperature is not None else {}
        if is_local_model(self.interpreter_model):
            # If enable_thinking is True, set max_interpretation_tokens to DEFAULT_MAX_INTERPRETATION_TOKENS_THINKING if it is lower than that
            if "enable_thinking" in config.llm.tokenizer_kwargs:
                if config.llm.tokenizer_kwargs["enable_thinking"]:
                    if config.llm.max_interpretation_tokens < DEFAULT_MAX_INTERPRETATION_TOKENS_THINKING:
                        config.llm.max_interpretation_tokens = DEFAULT_MAX_INTERPRETATION_TOKENS_THINKING

            raw_responses = get_local_completions(
                valid_prompts,
                model=self.interpreter_model,
                tokenizer_kwargs=config.llm.tokenizer_kwargs,
                max_tokens=config.llm.max_interpretation_tokens,
                llm_sampling_kwargs=llm_sampling_kwargs,
            )
            return [self._parse_interpretation(r) for r in raw_responses]

        # Use parallel threads to generate interpretations for API models
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers_interpretation) as executor:
            future_to_idx = {
                executor.submit(self._get_interpretation_openai, p, config): i
                for i, p in enumerate(valid_prompts)
            }

            iterator = tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(valid_prompts),
                desc="Generating interpretations",
            )

            ordered_interpretations = [None] * len(valid_prompts)
            for fut in iterator:
                idx = future_to_idx[fut]
                ordered_interpretations[idx] = fut.result()
            return ordered_interpretations

    def interpret_neurons(
        self,
        texts: List[str],
        activations: np.ndarray,
        neuron_indices: List[int],
        config: Optional[InterpretConfig] = None
    ) -> Dict[int, List[str]]:
        """Generate interpretations for multiple neurons with multiple candidates each."""
        config = config or InterpretConfig()
        interpretation_tasks = [
            (neuron_idx, candidate_idx)
            for neuron_idx in neuron_indices
            for candidate_idx in range(config.n_candidates)
        ]

        interpretations = {idx: [] for idx in neuron_indices}

        # Build prompts for every (neuron, candidate) pair
        prompts = []
        for neuron_idx, candidate_idx in interpretation_tasks:
            prompt = self._build_interpretation_prompt(
                texts=texts,
                activations=activations,
                neuron_idx=neuron_idx,
                candidate_idx=candidate_idx,
                config=config,
            )
            prompts.append(prompt)

        # Execute all valid prompts in a single batch (implementation-aware)
        generated_interpretations = self._execute_prompts(prompts, config)

        # Stitch responses back to their respective tasks / neurons
        interpretations_iterator = iter(generated_interpretations)
        for idx, (neuron_idx, _) in enumerate(interpretation_tasks):
            if prompts[idx] is None:
                interpretations[neuron_idx].append(None)
            else:
                interpretations[neuron_idx].append(next(interpretations_iterator))

        return interpretations

    def _compute_metrics(
        self,
        annotations: np.ndarray,
        labels: np.ndarray,
        activations: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics for a single interpretation.
        
        Args:
            annotations: Annotations computed by an LLM by applying a neuron's natural language interpretation to a set of examples
            labels: Binarized neuron activations (e.g. by setting the top-N activations to 1 and the zero-activations to 0) for the scored examples
            activations: Continuous neuron activations for the scored examples
            
        Returns:
            A dictionary containing the recall, precision, F1 score, and correlation for the interpretation
        """
        if not (1 in labels and 0 in labels):
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "correlation": 0.0}
            
        true_pos = np.mean(annotations[labels == 1])
        false_pos = np.mean(annotations[labels == 0])
        precision = 1 - false_pos
        
        f1 = 2 * true_pos * precision / (true_pos + precision) if (true_pos + precision) > 0 else 0.0
        correlation = np.corrcoef(activations, annotations)[0,1] if len(np.unique(annotations)) > 1 else 0.0
        
        return {
            "recall": true_pos,
            "precision": precision,
            "f1": f1,
            "correlation": correlation
        }

    def score_interpretations(
        self,
        texts: List[str],
        activations: np.ndarray,
        interpretations: Dict[int, List[str]],
        config: Optional[ScoringConfig] = None,
        show_progress: bool = True,
        **annotation_kwargs
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Score all interpretations for all neurons."""
        config = config or ScoringConfig()
        tasks = []
        scoring_info = {}

        for neuron_idx, neuron_interps in interpretations.items():
            formatted_examples = config.sampling_function(
                texts=texts,
                activations=activations,
                neuron_idx=neuron_idx,
                n_examples=config.n_examples,
                max_words_per_example=config.max_words_per_example,
                random_seed=neuron_idx,  # Deterministic seed based on neuron_idx
                **config.sampling_kwargs
            )
            
            eval_texts = formatted_examples["positive_texts"] + formatted_examples["negative_texts"]
            scoring_info[neuron_idx] = {
                'texts': eval_texts,
                'activations': formatted_examples["positive_activations"] + formatted_examples["negative_activations"],
                'binarized_activations': np.concatenate([
                    np.ones(len(formatted_examples["positive_texts"])),
                    np.zeros(len(formatted_examples["negative_texts"]))
                ])
            }

            for interp in neuron_interps:
                if interp is None:
                    continue
                for text in eval_texts:
                    tasks.append((text, interp))

        # Annotate all tasks
        cache_path = None if self.cache_name is None else os.path.join(CACHE_DIR, f"{self.cache_name}_interp-scoring.json")
        progress_desc = f"Scoring neuron interpretation fidelity ({len(interpretations)} neurons; {len(next(iter(interpretations.values())))} candidate interps per neuron; {config.n_examples} examples to score each interp)"
        annotations = annotate(
            tasks=tasks,
            cache_path=cache_path,
            n_workers=self.n_workers_annotation,
            show_progress=show_progress,
            model=self.annotator_model,
            progress_desc=progress_desc,
            **annotation_kwargs
        )

        # Compute metrics for all interpretations
        all_metrics = {}
        for neuron_idx, neuron_interps in interpretations.items():
            all_metrics[neuron_idx] = {}
            neuron_scoring_info = scoring_info[neuron_idx]

            for interp in neuron_interps:
                if interp is None:
                    all_metrics[neuron_idx][interp] = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "correlation": 0.0}
                    continue
                annot = [annotations[interp][text] for text in neuron_scoring_info['texts']]
                all_metrics[neuron_idx][interp] = self._compute_metrics(
                    annotations=np.array(annot),
                    labels=neuron_scoring_info['binarized_activations'],
                    activations=neuron_scoring_info['activations']
                )

        return all_metrics