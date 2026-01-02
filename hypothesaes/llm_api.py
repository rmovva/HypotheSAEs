"""LLM API utilities for HypotheSAEs."""

import os
import time
import openai
import google.genai
import anthropic

_CLIENT_OPENAI = None  # Module-level cache for the OpenAI client
_CLIENT_GOOGLE = None
_CLIENT_ANTR = None

"""
These model IDs point to the latest versions of the models as of 2025-05-04.
We point to a specific version for reproducibility, but feel free to update them as necessary.
Note that o-series models (o1, o1-mini, o3-mini) are also supported by get_completion().
We don't point these models to a specific version, so passing in these model names will use the latest version.

2025-05-04:
- Removed gpt-4 (deprecated by gpt-4o, will be removed from API soon)
- Added gpt-4.1 models (not used by HypotheSAEs paper, but potentially of interest)

2025-03-12:
- First version of this file: supports gpt-4o, gpt-4o-mini, gpt-4
"""
model_abbrev_to_id = {
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',

    "gpt4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",

    # Google Models
    "gemini-3.0-flash": "gemini-3-flash-preview",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.0-flash": "gemini-2.0-flash-001",

    # Anthropic Models
    "claude-opus": "claude-opus-4-5-20251101",
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    "claude-haiku": "claude-haiku-4-5-20251001"
}

model_to_provider = {
    'gpt': 'openai',
    'gemini': 'google',
    'claude': 'anthropic'
}

DEFAULT_MODEL = None

def _check_available_provider():
    """Check available provider based off API keys"""
    available = {}
    if os.environ.get('OPENAI_KEY_SAE') and '...' not in os.environ.get('OPENAI_KEY_SAE', ''):
        available['openai'] = True
    if os.environ.get('GOOGLE_KEY_SAE') and '...' not in os.environ.get('GOOGLE_KEY_SAE', ''):
        available['google'] = True
    if os.environ.get('ANTHROPIC_KEY_SAE') and '...' not in os.environ.get('ANTHROPIC_KEY_SAE', ''):
        available['anthropic'] = True
    return available

def _get_default_model():
    global DEFAULT_MODEL
    available = _check_available_provider()
    if 'openai' in available:
        DEFAULT_MODEL = "gpt-4.1-mini"
        return DEFAULT_MODEL
    elif 'google' in available:
        DEFAULT_MODEL = "gemini-2.5-flash-lite"
        return DEFAULT_MODEL
    elif 'anthropic' in available:
        DEFAULT_MODEL = "claude-sonnet"
        return DEFAULT_MODEL
    else:
        raise ValueError("No API keys found. Please set OPENAI_KEY_SAE, ANTHROPIC_KEY_SAE, or GOOGLE_KEY_SAE environment variables.")

def get_client(provider: str):
    """ 
    Get the API client for a specific model provider, initializing it if necessary and caching it.
    
    Args:
        provider: 'openai', 'google', 'anthropic'
    Returns:
        The appropriate API client
    
    Raises:
        Exception: If the required environment keys are not provided
    """
    global _CLIENT_OPENAI, _CLIENT_GOOGLE, _CLIENT_ANTR
    
    available = _check_available_provider()
    
    if provider == 'openai':
        if _CLIENT_OPENAI is not None:
            return _CLIENT_OPENAI
        if 'openai' not in available:
            raise ValueError("OPENAI_KEY_SAE not set. Cannot use OpenAI models.")
        oa_api_key = os.environ.get('OPENAI_KEY_SAE')
        _CLIENT_OPENAI = openai.OpenAI(api_key=oa_api_key)
        return _CLIENT_OPENAI
    
    elif provider == 'google':
        if _CLIENT_GOOGLE is not None:
            return _CLIENT_GOOGLE
        if 'google' not in available:
            raise ValueError("GOOGLE_KEY_SAE not set. Cannot use Google models.")
        g_api_key = os.environ.get('GOOGLE_KEY_SAE')
        _CLIENT_GOOGLE = google.genai.Client(api_key=g_api_key)
        return _CLIENT_GOOGLE
    
    elif provider == 'anthropic':
        if _CLIENT_ANTR is not None:
            return _CLIENT_ANTR
        if 'anthropic' not in available:
            raise ValueError("ANTHROPIC_KEY_SAE not set. Cannot use Anthropic models.")
        a_api_key = os.environ.get('ANTHROPIC_KEY_SAE')
        _CLIENT_ANTR = anthropic.Anthropic(api_key=a_api_key)
        return _CLIENT_ANTR
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs
) -> str:
    """
    Get completion from OpenAI API with retry logic and timeout.
    
    Args:
        prompt: The prompt to send
        model: Model to use
        max_retries: Maximum number of retries on rate limit
        backoff_factor: Factor to multiply backoff time by after each retry
        timeout: Timeout for the request
        **kwargs: Additional arguments to pass to the OpenAI API; max_tokens, temperature, etc.
    Returns:
        Generated completion text
    
    Raises:
        Exception: If all retries fail
    """
    select_provider = ''

    if model is None:
        model = _get_default_model()

    for pattern, provider in model_to_provider.items():
        if pattern in model:
            select_provider = provider

    if not select_provider:
        raise ValueError(f"Cannot infer provider for model '{model}'")


    client = get_client(provider=select_provider)
    model_id = model_abbrev_to_id.get(model, model)
    
    for attempt in range(max_retries):
        try:
            if select_provider == 'openai':
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                    **kwargs
                )
                return response.choices[0].message.content

            elif select_provider == 'google':
                response = client.models.generate_content(
                    model=model_id, 
                    contents=prompt,
                    config=google.genai.types.GenerateContentConfig(**kwargs)
                )
                return response.text
            
            elif select_provider == 'anthropic':
                response = client.messages.create(
                    model=model_id,
                    max_token=kwargs.pop('max_tokens', 1000),
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return response.content
            
        except (openai.RateLimitError, openai.APITimeoutError, google.genai.errors.APIError, anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            wait_time = timeout * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)