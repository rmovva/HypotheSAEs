import numpy as np
from hypothesaes import get_local_embeddings, train_sae
from hypothesaes import NeuronInterpreter, InterpretConfig, LLMConfig, SamplingConfig
from hypothesaes.annotate import annotate

from tests.sentences import BLUE_SENTENCES, RED_SENTENCES, ALL_TEST_SENTENCES

LOCAL_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM = "google/gemma-2-2b-it"

def test_local_interpretation():
    # Get local embeddings, train SAE, and get activations
    emb_dict = get_local_embeddings(ALL_TEST_SENTENCES, model=LOCAL_EMBEDDER, show_progress=False)
    embeddings = np.array([emb_dict[text] for text in ALL_TEST_SENTENCES])
    sae = train_sae(embeddings, M=2, K=1, n_epochs=2)
    activations = sae.get_activations(embeddings)

    interpreter = NeuronInterpreter(
        interpreter_model=LOCAL_LLM,
        annotator_model=LOCAL_LLM,
        n_workers_interpretation=1,
        n_workers_annotation=1,
    )
    config = InterpretConfig(
        sampling=SamplingConfig(n_examples=4),
        llm=LLMConfig(max_interpretation_tokens=50),
    )
    results = interpreter.interpret_neurons(ALL_TEST_SENTENCES, activations, [0, 1], config)
    print(results)
    for idx in [0, 1]:
        assert idx in results
        assert isinstance(results[idx][0], str)
        assert len(results[idx][0]) > 0

def test_local_annotation():
    blue_concept = "mentions something that is blue"
    red_concept = "mentions something that is red"
    positive_tasks = [(text, blue_concept) for text in BLUE_SENTENCES] + [(text, red_concept) for text in RED_SENTENCES]
    negative_tasks = [(text, blue_concept) for text in RED_SENTENCES] + [(text, red_concept) for text in BLUE_SENTENCES]
    results = annotate(positive_tasks + negative_tasks, model=LOCAL_LLM, show_progress=True, n_workers=1)
    
    # Calculate precision and recall for each concept
    for concept, concept_dict in results.items():
        if concept == blue_concept:
            tp = sum(1 for text in BLUE_SENTENCES if concept_dict.get(text, 0) == 1)
            fp = sum(1 for text in RED_SENTENCES if concept_dict.get(text, 0) == 1)
            fn = sum(1 for text in BLUE_SENTENCES if concept_dict.get(text, 0) == 0)
        else:  # red_concept
            tp = sum(1 for text in RED_SENTENCES if concept_dict.get(text, 0) == 1)
            fp = sum(1 for text in BLUE_SENTENCES if concept_dict.get(text, 0) == 1)
            fn = sum(1 for text in RED_SENTENCES if concept_dict.get(text, 0) == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"{concept.split()[1]} concept - P: {precision:.3f}, R: {recall:.3f}")

    for concept_dict in results.values():
        for val in concept_dict.values():
            assert val in (0, 1)
