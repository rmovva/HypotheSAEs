"""HypotheSAEs: SAEs for hypothesis generation."""

# Version information
__version__ = "0.0.3"

# Import key functions and classes to expose at the package level
from .quickstart import (
    train_sae,
    interpret_sae,
    generate_hypotheses,
    evaluate_hypotheses
)

from .sae import (
    SparseAutoencoder,
    load_model,
    get_multiple_sae_activations
)

from .embedding import (
    get_openai_embeddings,
    get_local_embeddings
)

from .interpret_neurons import (
    NeuronInterpreter,
    InterpretConfig,
    ScoringConfig,
    LLMConfig,
    SamplingConfig
)

from .select_neurons import select_neurons

from .evaluation import score_hypotheses

from .annotate import annotate_texts_with_concepts

from .utils import get_text_for_printing

# Define what gets imported with "from hypothesaes import *"
__all__ = [
    # Main workflow functions
    "train_sae",
    "interpret_sae", 
    "generate_hypotheses", 
    "evaluate_hypotheses",
    
    # Core classes
    "SparseAutoencoder",
    "load_model",
    "get_multiple_sae_activations",
    
    # Embedding functions
    "get_openai_embeddings",
    "get_local_embeddings",
    
    # Interpretation classes
    "NeuronInterpreter",
    "InterpretConfig",
    "ScoringConfig",
    "LLMConfig",
    "SamplingConfig",
    
    # Selection and evaluation
    "select_neurons",
    "score_hypotheses",
    "annotate_texts_with_concepts",
    
    # Utilities
    "get_text_for_printing"
]