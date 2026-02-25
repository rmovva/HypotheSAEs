"""LLM API utilities for HypotheSAEs."""

import os
import time
import logging
from typing import Any, Dict, List, Optional

import openai

# Suppress INFO-level request logs from OpenAI/httpx to keep notebook output clean.
for _logger_name in ("openai", "openai._client", "openai._base_client", "httpx", "httpcore"):
    logger = logging.getLogger(_logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

_CLIENT_OPENAI = {}  # Cache keyed by (api_key, base_url)

"""
These model IDs point to the latest versions of the models.
We point to specific versions only when needed; otherwise we use the latest.
"""
model_abbrev_to_id = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt4.1": "gpt-4.1",
    "gpt-4.1": "gpt-4.1",
    "gpt4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt4.1-nano": "gpt-4.1-nano",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "gpt5.2": "gpt-5.2",
    "gpt-5.2": "gpt-5.2",
    "gpt5.2-chat-latest": "gpt-5.2-chat-latest",
    "gpt-5.2-chat-latest": "gpt-5.2-chat-latest",
    "gpt5.2-pro": "gpt-5.2-pro",
    "gpt-5.2-pro": "gpt-5.2-pro",
    "gpt5.2-codex": "gpt-5.2-codex",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt5-mini": "gpt-5-mini",
    "gpt-5-mini": "gpt-5-mini",
    "gpt5-nano": "gpt-5-nano",
    "gpt-5-nano": "gpt-5-nano",
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",
}

DEFAULT_MODEL = "gpt-5-mini"


def _normalize_message_content(content: Any) -> Any:
    """Convert Chat Completions-style content items to Responses API items."""
    if not isinstance(content, list):
        return content

    normalized = []
    for item in content:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        item_type = item.get("type")
        if item_type == "text":
            normalized.append({"type": "input_text", "text": item.get("text", "")})
            continue

        normalized.append(item)
    return normalized


def _normalize_messages_for_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of chat messages to Responses API input items."""
    normalized_messages = []
    for message in messages:
        normalized = dict(message)
        normalized["content"] = _normalize_message_content(message.get("content"))
        normalized_messages.append(normalized)
    return normalized_messages


def _get_field(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _extract_output_text(response: Any) -> str:
    """Extract assistant text from a Responses API response object."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    output = getattr(response, "output", None)
    if not output:
        output = response.get("output") if isinstance(response, dict) else None
    if not output:
        output = []

    for item in output:
        item_type = _get_field(item, "type")
        if item_type == "output_text":
            text = _get_field(item, "text")
            if text:
                return text
        if item_type == "message":
            content_items = _get_field(item, "content") or []
            for content in content_items:
                content_type = _get_field(content, "type")
                if content_type in ("output_text", "text"):
                    text = _get_field(content, "text")
                    if text:
                        return text
                if content_type is None:
                    text = _get_field(content, "text")
                    if text:
                        return text

    # Fallback for endpoints that still return chat/completions-like payloads.
    choices = _get_field(response, "choices")
    if choices:
        first_choice = choices[0]
        message = _get_field(first_choice, "message")
        if message:
            text = _get_field(message, "content")
            if text:
                return text
        text = _get_field(first_choice, "text")
        if text:
            return text

    return ""


def normalize_llm_kwargs(
    llm_kwargs: Optional[Dict[str, Any]] = None,
    *,
    default_verbosity: Optional[str] = None,
    default_reasoning_effort: Optional[str] = None,
    default_timeout: Optional[float] = None,
    default_max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply default LLM kwargs without overriding explicit user settings."""
    resolved = dict(llm_kwargs or {})
    if default_verbosity is not None and "verbosity" not in resolved and "text" not in resolved:
        resolved["verbosity"] = default_verbosity
    if (
        default_reasoning_effort is not None
        and "reasoning" not in resolved
        and "reasoning_effort" not in resolved
    ):
        resolved["reasoning_effort"] = default_reasoning_effort
    if default_timeout is not None and "timeout" not in resolved:
        resolved["timeout"] = default_timeout
    if default_max_output_tokens is not None and "max_output_tokens" not in resolved:
        resolved["max_output_tokens"] = default_max_output_tokens
    return resolved

def get_client():
    """
    Get an OpenAI-compatible client.

    - OPENAI_KEY_SAE: required API key (OpenAI cloud or dummy for vLLM)
    - OPENAI_BASE_URL: optional base URL to point at a vLLM/OpenAI-compatible server
      (fallback is OpenAI cloud)
    """
    global _CLIENT_OPENAI

    api_key = os.environ.get("OPENAI_KEY_SAE")
    if api_key is None or "..." in api_key:
        raise ValueError("Please set the OPENAI_KEY_SAE environment variable before using functions which require the OpenAI API.")

    base_url = os.environ.get("OPENAI_BASE_URL")
    cache_key = (api_key, base_url or "__openai_default__")
    if cache_key in _CLIENT_OPENAI:
        return _CLIENT_OPENAI[cache_key]

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    _CLIENT_OPENAI[cache_key] = openai.OpenAI(**client_kwargs)
    return _CLIENT_OPENAI[cache_key]

def get_completion(
    prompt: Optional[str] = None,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL,
    timeout: Optional[float] = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs
) -> str:
    """
    Get completion from OpenAI Responses API with retry logic and timeout.
    
    Args:
        prompt: Raw text prompt (used when ``messages`` is None)
        messages: Optional list of chat messages
        model: Model to use
        max_retries: Maximum number of retries on rate limit
        backoff_factor: Factor to multiply backoff time by after each retry
        timeout: Optional timeout for the request
        **kwargs: Additional arguments to pass to the Responses API; max_output_tokens, reasoning, text, etc.
    Returns:
        Generated completion text
    
    Raises:
        Exception: If all retries fail
    """
    if prompt is None and messages is None:
        raise ValueError("Either prompt or messages must be provided to get_completion()")

    client = get_client()
    model_id = model_abbrev_to_id.get(model, model)

    max_output_tokens = kwargs.pop("max_output_tokens", None)
    if max_output_tokens is None:
        max_output_tokens = kwargs.pop("max_completion_tokens", None)
    if max_output_tokens is None:
        max_output_tokens = kwargs.pop("max_tokens", None)

    verbosity = kwargs.pop("verbosity", None)
    if verbosity is not None:
        text_payload = dict(kwargs.pop("text", {}) or {})
        text_payload["verbosity"] = verbosity
        kwargs["text"] = text_payload

    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort is not None:
        reasoning_payload = dict(kwargs.pop("reasoning", {}) or {})
        reasoning_payload["effort"] = reasoning_effort
        kwargs["reasoning"] = reasoning_payload

    request_input = _normalize_messages_for_responses(messages) if messages is not None else prompt

    # GPT-5 models currently reject extremely small limits.
    if max_output_tokens is not None and max_output_tokens < 16 and model_id.startswith("gpt-5"):
        max_output_tokens = 16

    base_wait = timeout if timeout is not None else 1.0
    for attempt in range(max_retries):
        try:
            request_kwargs = dict(kwargs)
            if max_output_tokens is not None:
                request_kwargs["max_output_tokens"] = max_output_tokens
            if timeout is not None:
                request_kwargs["timeout"] = timeout

            response = client.responses.create(
                model=model_id,
                input=request_input,
                **request_kwargs
            )
            return _extract_output_text(response)
            
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            wait_time = base_wait * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)
