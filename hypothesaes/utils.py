"""Core utilities for HypotheSAEs."""

import os
import json
from typing import List, Optional
import openai
from pathlib import Path
import time
import tiktoken

# OpenAI client setup
api_key = os.environ.get('OPENAI_KEY_SAE') 
if api_key is None:
    raise ValueError("Please set the OPENAI_KEY_SAE environment variable before using HypotheSAEs.")

client = openai.OpenAI(
    api_key=api_key,
)

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

def get_completion(
    prompt: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    request_timeout: float = 20.0,
    **kwargs
) -> str:
    """
    Get completion from OpenAI API with retry logic and request timeout.
    
    Args:
        prompt: The prompt to send
        model: Model to use
        max_retries: Maximum number of retries on rate limit
        backoff_factor: Factor to multiply backoff time by after each retry
        request_timeout: Timeout for the request
        **kwargs: Additional arguments to pass to the OpenAI API
    Returns:
        Generated completion text
    
    Raises:
        Exception: If all retries fail
    """
    model_id = model_abbrev_to_id.get(model, model)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                timeout=request_timeout,
                **kwargs
            )
            response_text = response.choices[0].message.content
            if len(response_text) == 0:
                print(f"Empty response from OpenAI API, finish reason: {response.choices[0].finish_reason}")
            return response_text
            
        except openai.APITimeoutError as e:
            if attempt == max_retries - 1:  # Last attempt
                raise TimeoutError(f"API request timed out after {request_timeout}s")
            if attempt > 0:
                wait_time = request_timeout * (backoff_factor ** attempt)
                print(f"API timeout, retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            # Exponential backoff
            wait_time = (request_timeout * (backoff_factor ** attempt))
            print(f"API error, retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)

def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{prompt_name}.txt"
    try:
        with open(prompt_path) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {prompt_path}; please ensure it's in the src/prompts/ directory")

def truncate_text(
    text: str,
    max_words: Optional[int] = None,
    max_chars: Optional[int] = None,
    max_tokens: Optional[int] = None,
    truncation_message: str = "[... rest of text is truncated]"
) -> str:
    """
    Truncate text based on words, characters, or tokens.
    
    Args:
        text: Input text to truncate
        max_words: Maximum number of words
        max_chars: Maximum number of characters
        max_tokens: Maximum number of tokens (using tiktoken)
    
    Returns:
        Truncated text with indicator if truncated
    """
    if all(x is None for x in [max_words, max_chars, max_tokens]):
        return text
    
    if text.endswith(truncation_message):
        return text
    truncated = text
    
    if max_words is not None:
        words = text.split()
        if len(words) > max_words:
            truncated = ' '.join(words[:max_words])
            
    if max_chars is not None:
        if len(truncated) > max_chars:
            truncated = truncated[:max_chars]
            
    if max_tokens is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(truncated)
        if len(tokens) > max_tokens:
            truncated = enc.decode(tokens[:max_tokens])
            
    if truncated != text:
        truncated += truncation_message
        
    return truncated

def get_text_for_printing(text: str, max_chars: int = 128) -> str:
    """Truncate and remove newlines from a string."""
    return truncate_text(text, max_chars=max_chars).replace('\n', ' ')

def filter_invalid_texts(texts: List[str]) -> List[str]:
    """Filter out None values and empty strings from a list of texts.
    
    Args:
        texts: List of text strings, potentially containing None or empty strings
        
    Returns:
        Filtered list with None values and empty strings removed
    """
    original_count = len(texts)
    filtered_texts = [text for text in texts if text is not None and text != ""]
    filtered_count = original_count - len(filtered_texts)
    
    if filtered_count > 0:
        print(f"Warning: ignoring {filtered_count} items which are None or empty strings")
    
    return filtered_texts