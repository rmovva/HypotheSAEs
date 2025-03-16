"""LLM API utilities for HypotheSAEs."""

import os
import time
import openai

"""
These model IDs point to the latest versions of the models as of 2025-03-12.
We point to a specific version for reproducibility, but feel free to update them as necessary.
Note that o1, o1-mini, o3-mini are also supported by get_completion().
We don't point these models to a specific version, so passing in these model names will use the latest version.
"""
model_abbrev_to_id = {
    'gpt4': 'gpt-4-0125-preview',
    'gpt-4': 'gpt-4-0125-preview',
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
}

def get_client():
    """Get the OpenAI client, initializing it if necessary."""
    api_key = os.environ.get('OPENAI_KEY_SAE')
    if api_key is None or '...' in api_key:
        raise ValueError("Please set the OPENAI_KEY_SAE environment variable before using functions which require the OpenAI API.")
    
    return openai.OpenAI(api_key=api_key)

def get_completion(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 1000,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> str:
    """Get completion from OpenAI API with retry logic."""
    client = get_client()
    model_id = model_abbrev_to_id.get(model, model)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            # Exponential backoff
            wait_time = timeout * (backoff_factor ** attempt)
            print(f"API timeout, retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)