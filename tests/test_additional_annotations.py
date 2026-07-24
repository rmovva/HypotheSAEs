import os

import numpy as np
import pytest

from hypothesaes.annotate import annotate, parse_completion_json

from .sentences import BLUE_SENTENCES, RED_SENTENCES

if os.getenv('OPENAI_KEY_SAE') is None or os.getenv('OPENAI_KEY_SAE') == '...':
    raise ValueError("Please set the OPENAI_KEY_SAE environment variable before running tests.")

def _test_annotation(annotator_model):
    blue_concept = "contains words associated with the color blue"
    tasks = [(BLUE_SENTENCES[0], blue_concept), (RED_SENTENCES[0], blue_concept)]
    annotations = annotate(
        tasks,
        model=annotator_model,
        show_progress=False,
        temperature=0.0,
        parse_fn=parse_completion_json,
        annotate_prompt_name='annotate-user-json',
        system_prompt_name='annotate-system-json'
    )
    print(annotations)
    assert blue_concept in annotations
    assert BLUE_SENTENCES[0] in annotations[blue_concept]
    assert RED_SENTENCES[0] in annotations[blue_concept]
    assert annotations[blue_concept][BLUE_SENTENCES[0]] in (0, 1)
    assert annotations[blue_concept][RED_SENTENCES[0]] in (0, 1)

def test_openai_annotation():
    _test_annotation("gpt-4.1-mini")