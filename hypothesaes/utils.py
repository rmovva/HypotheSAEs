"""Core utilities for HypotheSAEs."""

import os
import json
from typing import List, Optional
from pathlib import Path
import time
import tiktoken


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{prompt_name}.txt"
    try:
        with open(prompt_path) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {prompt_path}; please ensure it's in the hypothesaes/prompts/ directory")

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
    filtered_texts = [text for text in texts if text is not None and len(str(text).strip()) > 0]
    filtered_count = original_count - len(filtered_texts)
    
    if filtered_count > 0:
        print(f"Warning: ignoring {filtered_count} items which are None or empty strings")
    
    return filtered_texts

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to a JSON file, creating directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)