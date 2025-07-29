import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pandas as pd
import numpy as np

import json
import pickle

from hypothesaes.interpret_neurons import NeuronInterpreter, ScoringConfig, sample_percentile_bins

base_dir = os.path.join('/nas/ucb/rmovva/data/hypothesaes')
train_df = pd.read_json(os.path.join(base_dir, 'hypothesis-generation-data', 'yelp', "train-200K.json"), lines=True)
texts = train_df["text"].tolist()
all_model_interpretations = json.load(open(os.path.join(base_dir, 'local_llm_experiments', 'results', 'interpretations_1024_32.json')))
activations = np.load(os.path.join(base_dir, 'local_llm_experiments', 'checkpoints', 'activations_1024_32.npy'))

interpreter = NeuronInterpreter(annotator_model="gpt-4.1-mini", n_workers_annotation=200, cache_name="v1-test-autointerp-yelp-qwen")
scoring_config = ScoringConfig(
    n_examples=100,
    sampling_function=sample_percentile_bins,
    sampling_kwargs={
        "high_percentile": (80, 100),
        "low_percentile": None,
    },
)

fidelities = {}
for model, neuron_interpretations in all_model_interpretations.items():
    neuron_interpretations = {int(k): v for k, v in neuron_interpretations.items()}
    print(f"Scoring {model}")
    fidelities[model] = interpreter.score_interpretations(
        texts=texts,
        activations=activations,
        interpretations=neuron_interpretations,
        config=scoring_config,
        annotate_prompt_name="annotate-simple", # Zero-shot annotation prompt
    )

with open(os.path.join(base_dir, 'local_llm_experiments', 'results', 'fidelities_1024_32.pkl'), "wb") as f:
    pickle.dump(fidelities, f)