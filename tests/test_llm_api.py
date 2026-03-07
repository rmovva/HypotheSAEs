import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


_LLM_API_PATH = Path(__file__).resolve().parents[1] / "hypothesaes" / "llm_api.py"
_SPEC = importlib.util.spec_from_file_location("hypothesaes_llm_api_for_tests", _LLM_API_PATH)
llm_api = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(llm_api)


def _clear_client_cache():
    llm_api._CLIENT_OPENAI.clear()


def test_resolve_api_key_requires_key_for_default_openai():
    with patch.dict(llm_api.os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_KEY_SAE"):
            llm_api._resolve_api_key(None)


def test_resolve_api_key_allows_local_base_url_without_key():
    with patch.dict(llm_api.os.environ, {}, clear=True):
        api_key = llm_api._resolve_api_key("http://127.0.0.1:8000/v1")

    assert api_key == llm_api.LOCAL_OPENAI_API_KEY_PLACEHOLDER


def test_get_client_uses_placeholder_key_for_local_base_url():
    captured = {}

    class DummyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    _clear_client_cache()
    with patch.dict(llm_api.os.environ, {"OPENAI_BASE_URL": "http://127.0.0.1:8000/v1"}, clear=True):
        with patch.object(llm_api.openai, "OpenAI", DummyClient):
            llm_api.get_client()

    assert captured["api_key"] == llm_api.LOCAL_OPENAI_API_KEY_PLACEHOLDER
    assert captured["base_url"] == "http://127.0.0.1:8000/v1"
    _clear_client_cache()


def test_get_client_requires_key_for_openai_base_url():
    _clear_client_cache()
    with patch.dict(llm_api.os.environ, {"OPENAI_BASE_URL": "https://api.openai.com/v1"}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_KEY_SAE"):
            llm_api.get_client()

    _clear_client_cache()
