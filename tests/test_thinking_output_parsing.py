from hypothesaes.annotate import parse_completion
from hypothesaes.interpret_neurons import NeuronInterpreter


def test_parse_completion_after_think_tags_yes():
    completion = "<think>reasoning...</think>\n\nyes"
    assert parse_completion(completion) == 1


def test_parse_completion_after_think_tags_no():
    completion = "<think>reasoning...</think>\n\nno"
    assert parse_completion(completion) == 0


def test_parse_interpretation_after_think_tags():
    interpreter = NeuronInterpreter()
    response = "<think>reasoning...</think>\n\n- uses words related to blue"
    assert interpreter._parse_interpretation(response) == "uses words related to blue"


def test_parse_interpretation_incomplete_think_returns_none():
    interpreter = NeuronInterpreter()
    response = "<think>unfinished reasoning"
    assert interpreter._parse_interpretation(response) is None
