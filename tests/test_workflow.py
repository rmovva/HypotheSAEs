import os
import pytest
import numpy as np
import torch
from pathlib import Path
import hypothesaes
from hypothesaes import (
    get_openai_embeddings,
    get_local_embeddings,
    train_sae,
    interpret_sae,
    generate_hypotheses,
    evaluate_hypotheses
)
from hypothesaes.sae import get_sae_checkpoint_name, load_model

from hypothesaes.llm_api import get_completion

if os.getenv('OPENAI_KEY_SAE') is None or os.getenv('OPENAI_KEY_SAE') == '...':
    raise ValueError("Please set the OPENAI_KEY_SAE environment variable before running tests.")

LOCAL_MODEL_TESTING = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL_TESTING = "text-embedding-3-small"

# Create a dummy dataset with blue and red objects
BLUE_OBJECTS = [
    "The ocean waves crashed onto the shore, reflecting azure light.",
    "The cloudless sky stretched endlessly, a perfect cerulean.",
    "Wild blue delphiniums swayed in the summer breeze.",
    "The deep sea stretched to the horizon, dark and mysterious.",
    "Her room was painted in a calming aqua shade.",
    "The recycling bins lined up along the curb, all in navy.",
    "A blue jay hopped between branches in the garden.",
    "A massive whale breached the surface, gleaming indigo.",
    "A morpho butterfly's wings shimmered in the sunlight.",
    "The turquoise pool beckoned on a hot day.",
    "The officer's navy uniform was perfectly pressed.",
    "Cornflowers dotted the wildflower meadow with sapphire.",
    "The sapphire sparkled under the display lights.",
    "The blueprints were spread across the desk.",
    "The aged Roquefort cheese had distinctive blue veins.",
    "A spruce tree towered over the forest, tinted cobalt.",
    "Her eyes matched the color of a mountain lake.",
    "The company logo featured their signature azure shade.",
    "Bachelor's buttons bloomed with periwinkle petals.",
    "The lake mirrored the cerulean sky above."
]

RED_OBJECTS = [
    "The firetruck's scarlet paint gleamed in the sun.",
    "The stop sign commanded attention at the corner.",
    "She drizzled marinara sauce, rich and crimson.",
    "Ripe tomatoes blushed in the California sun.",
    "Students clutched their ruby-colored cups at the party.",
    "Her lipstick was a bold scarlet shade.",
    "A cardinal's bright feathers caught the morning light.",
    "Long-stemmed roses filled the crystal vase.",
    "The matador's cape swirled with crimson flair.",
    "Warm blood flowed through her beating heart.",
    "Bell peppers ripened to a brilliant burgundy.",
    "The recording light pulsed in warning crimson.",
    "Fresh strawberries dotted the garden patch.",
    "Her cheeks flushed with a rosy glow.",
    "The sunset painted everything in vermilion.",
    "The red panda's russet fur gleamed as it ate.",
    "Maple leaves turned scarlet in autumn's embrace.",
    "His burgundy tie added elegance to the suit.",
    "Aid workers wore their distinctive crimson vests.",
    "The warning light flashed an urgent ruby glow."
]

ALL_SENTENCES = BLUE_OBJECTS + RED_OBJECTS
# Labels: 0 for blue objects, 1 for red objects
LABELS = [0] * len(BLUE_OBJECTS) + [1] * len(RED_OBJECTS)

@pytest.fixture(scope="module")
def test_data():
    """Return test data and precomputed local embeddings to use across all tests."""
    sentences = ALL_SENTENCES
    labels = LABELS

    # Compute local embeddings (MiniLM)
    local_emb_dict = get_local_embeddings(texts=sentences, model=LOCAL_MODEL_TESTING, show_progress=False)
    local_embeddings = np.array([local_emb_dict[text] for text in sentences])

    return {
        "sentences": sentences,
        "labels": labels,
        "local_embeddings": local_embeddings,
    }

def test_openai_client():
    """Test the OpenAI client."""
    test_completion = get_completion(
        prompt="Reply with 'Hello, world!'",
        model="gpt-4o-mini",
        max_tokens=1,
    )
    assert test_completion is not None
    assert len(test_completion) > 0

def test_compute_openai_embeddings(test_data):
    """Test computing embeddings using OpenAI models."""
    sentences = test_data["sentences"]
    openai_emb_dict = get_openai_embeddings(texts=sentences, model=OPENAI_MODEL_TESTING, show_progress=False, n_workers=1)
    openai_embeddings = np.array([openai_emb_dict[text] for text in sentences])
    assert openai_embeddings.shape == (len(ALL_SENTENCES), 1536), f"OpenAI embeddings shape is {openai_embeddings.shape}, expected ({len(ALL_SENTENCES)}, 1536)"

def test_compute_local_embeddings(test_data):
    """Test local embeddings shape."""
    local_embeddings = test_data["local_embeddings"]
    assert local_embeddings.shape == (len(ALL_SENTENCES), 384), f"Local embeddings shape is {local_embeddings.shape}, expected ({len(ALL_SENTENCES)}, 384)"

def test_train_sae(test_data):
    """Test training, saving, and loading SAEs with different configurations."""
    M, K = 2, 1
    checkpoint_dir = "./"
    _ = train_sae(test_data["local_embeddings"], M, K, n_epochs=3, checkpoint_dir=checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, get_sae_checkpoint_name(M, K))
    assert os.path.exists(checkpoint_path)
    _ = load_model(checkpoint_path)
    os.remove(checkpoint_path)

def test_train_matryoshka_sae(test_data):
    """Test training a Matryoshka SAE (with multiple prefix lengths)."""
    matryoshka_prefix_lengths = [2, 4]
    sae = train_sae(embeddings=test_data["local_embeddings"], M=4, K=1, matryoshka_prefix_lengths=matryoshka_prefix_lengths, n_epochs=3)
    assert sae.prefix_lengths == matryoshka_prefix_lengths

def test_interpret_sae(test_data):
    """Test interpreting neurons from trained SAEs."""
    sentences = test_data["sentences"]
    embeddings = test_data["local_embeddings"]
    sae_small = train_sae(embeddings=embeddings, M=2, K=1, n_epochs=3)
    sae_large = train_sae(embeddings=embeddings, M=4, K=2, n_epochs=3)

    # Test interpret_sae with both SAEs
    interpretations = interpret_sae(
        texts=sentences,
        embeddings=embeddings,
        sae=[sae_small, sae_large],
        n_random_neurons=2,
        interpreter_model="gpt-4o",  # Use smaller model for testing
        annotator_model="gpt-4o-mini",
        n_examples_for_interpretation=10,  # Small number for testing
        print_examples_n=3
    )
    assert len(interpretations) > 0
    assert "neuron_idx" in interpretations.columns
    assert "interpretation" in interpretations.columns

def test_generate_and_evaluate_hypotheses(test_data):
    """Test generating and evaluating hypotheses."""
    sentences = test_data["sentences"]
    labels = test_data["labels"]
    embeddings = test_data["local_embeddings"]
    sae_small = train_sae(embeddings=embeddings, M=2, K=1, n_epochs=3)
    sae_large = train_sae(embeddings=embeddings, M=4, K=2, n_epochs=3)

    # Generate hypotheses
    hypotheses_df = generate_hypotheses(
        texts=sentences,
        labels=labels,
        embeddings=embeddings,
        sae=[sae_small, sae_large],
        classification=True,
        n_selected_neurons=2,  # Small number for testing
        interpreter_model="gpt-4o",
        annotator_model="gpt-4o-mini",
        n_examples_for_interpretation=10,
        n_scoring_examples=10,
        n_workers_annotation=100
    )
    
    # Verify hypotheses structure
    assert len(hypotheses_df) > 0
    assert "neuron_idx" in hypotheses_df.columns
    assert "interpretation" in hypotheses_df.columns
    print(hypotheses_df)

    # Evaluate hypotheses
    metrics, evaluation_df = evaluate_hypotheses(
        hypotheses_df=hypotheses_df,
        texts=sentences,
        labels=labels,
        annotator_model="gpt-4o-mini",
        n_workers_annotation=100
    )
    
    # Verify evaluation structure
    assert len(evaluation_df) > 0
    assert "hypothesis" in evaluation_df.columns
    assert "regression_coef" in evaluation_df.columns 
    print(evaluation_df)