import os

import numpy as np
import pytest

from hypothesaes.annotate import annotate
from hypothesaes.interpret_neurons import InterpretConfig, LLMConfig, NeuronInterpreter, SamplingConfig
from hypothesaes.llm_api import get_completion

from .sentences import BLUE_SENTENCES, RED_SENTENCES


def _require_local_openai_test() -> None:
    if os.getenv("RUN_LOCAL_OPENAI_TEST") != "1":
        pytest.skip("Set RUN_LOCAL_OPENAI_TEST=1 to run local OpenAI-compatible endpoint tests.")


def _with_local_base_url():
    class _BaseUrlGuard:
        def __enter__(self):
            self.previous = os.environ.get("OPENAI_BASE_URL")
            os.environ["OPENAI_BASE_URL"] = os.getenv("LOCAL_OPENAI_BASE_URL", "http://0.0.0.0:8000/v1")
            return os.getenv("LOCAL_OPENAI_MODEL", "Qwen/Qwen3-8B")

        def __exit__(self, exc_type, exc, tb):
            if self.previous is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = self.previous

    return _BaseUrlGuard()


def test_local_openai_completion():
    _require_local_openai_test()
    with _with_local_base_url() as local_model:
        completion = get_completion(
            prompt="Reply with exactly: hello",
            model=local_model,
            max_output_tokens=16,
            temperature=0.0,
        )
    assert completion is not None
    assert len(completion.strip()) > 0


def test_local_openai_annotation_and_interpretation():
    _require_local_openai_test()

    with _with_local_base_url() as local_model:
        blue_concept = "contains words associated with the color blue"
        tasks = [(BLUE_SENTENCES[0], blue_concept), (RED_SENTENCES[0], blue_concept)]
        annotations = annotate(
            tasks,
            model=local_model,
            show_progress=False,
            max_output_tokens=16,
            temperature=0.0,
        )
        assert blue_concept in annotations
        assert BLUE_SENTENCES[0] in annotations[blue_concept]
        assert RED_SENTENCES[0] in annotations[blue_concept]
        assert annotations[blue_concept][BLUE_SENTENCES[0]] in (0, 1)
        assert annotations[blue_concept][RED_SENTENCES[0]] in (0, 1)

        texts = BLUE_SENTENCES[:8] + RED_SENTENCES[:8]
        activations = np.stack(
            [
                np.concatenate([np.ones(8), np.zeros(8)]),
                np.concatenate([np.zeros(8), np.ones(8)]),
            ],
            axis=1,
        )
        interpreter = NeuronInterpreter(interpreter_model=local_model)
        config = InterpretConfig(
            sampling=SamplingConfig(n_examples=10),
            llm=LLMConfig(max_output_tokens=32, temperature=0.0),
        )
        results = interpreter.interpret_neurons(texts, activations, neuron_indices=[0], config=config)
        assert 0 in results
        assert isinstance(results[0][0], str)
        assert len(results[0][0]) > 0
