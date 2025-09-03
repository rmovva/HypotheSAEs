"""Utilities for computing text embeddings."""

import numpy as np
from typing import List, Optional, Dict
import concurrent.futures
from tqdm.auto import tqdm
import tiktoken
import os
import time
from pathlib import Path
import glob
import torch
import openai
from .utils import filter_invalid_texts
from sentence_transformers import SentenceTransformer
import torch
import gc


# Use environment variable for cache dir if set, otherwise use default
CACHE_DIR = os.getenv('EMB_CACHE_DIR') or os.path.join(Path(__file__).parent.parent, 'emb_cache')

def _embed_batch_openai(
        batch: List[str], 
        model: str, 
        client,
        max_tokens: int = 8192, 
        max_retries: int = 3, 
        backoff_factor: float = 3.0,
        timeout: float = 10.0
) -> List[List[float]]:
    """Helper function for batch embedding using OpenAI API."""
    # Truncate texts to max tokens
    enc = tiktoken.get_encoding("cl100k_base")  # encoding for OpenAI text-embedding models
    truncated_batch = []
    for text in batch:
        tokens = enc.encode(text.strip())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
        truncated_batch.append(text)
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=truncated_batch,
                model=model,
                timeout=timeout
            )
            return [data.embedding for data in response.data]
            
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            wait_time = timeout * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)


def load_embedding_cache(cache_name: str) -> dict:
    """Load cached embeddings from chunked files."""
    if not cache_name:
        return {}
    
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return {}
    
    text2embedding = {}
    chunk_files = sorted(glob.glob(f"{cache_dir}/chunk_*.npy"))
    
    start_time = time.time()
    for chunk_file in tqdm(chunk_files, desc="Loading embedding chunks"):
        # Each chunk file contains a list of (text, embedding) tuples
        chunk_data = np.load(chunk_file, allow_pickle=True)
        for text, emb in chunk_data:
            text2embedding[text] = emb
            
    load_time = time.time() - start_time
    print(f"Loaded {len(text2embedding)} embeddings in {load_time:.1f}s")
            
    return text2embedding


def update_embedding_cache(cache_name: str, text2embedding: dict, chunk_size: int = 50000) -> None:
    """Update cache files in chunks."""
    if not cache_name:
        return
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Convert to list of (text, embedding) tuples for storage
    items = list(text2embedding.items())
    
    for i in range(0, len(items), chunk_size):
        chunk_items = items[i:i + chunk_size]
        chunk_num = i // chunk_size
        chunk_path = f"{cache_dir}/chunk_{chunk_num:03d}.npy"
        np.save(chunk_path, np.array(chunk_items, dtype=object))
        

def _get_next_chunk_index(cache_name: str) -> int:
    """Determine the next available chunk index for a cache."""
    if not cache_name:
        return 0
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return 0
        
    chunk_files = glob.glob(f"{cache_dir}/chunk_*.npy")
    if not chunk_files:
        return 0
        
    indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in chunk_files]
    return max(indices) + 1
    
def _save_embedding_chunk(cache_name: str, chunk_embeddings: dict, chunk_idx: int) -> int:
    """Save a chunk of embeddings to disk."""
    if not cache_name or not chunk_embeddings:
        return chunk_idx
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    chunk_path = f"{cache_dir}/chunk_{chunk_idx:03d}.npy"
    chunk_items = list(chunk_embeddings.items())
    np.save(chunk_path, np.array(chunk_items, dtype=object))
    print(f"Saved {len(chunk_items)} embeddings to {chunk_path}")
    
    return chunk_idx + 1

def get_openai_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 256,
    n_workers: int = 5,
    cache_name: Optional[str] = None,
    show_progress: bool = True,
    chunk_size: int = 50000,
    timeout: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Get embeddings using OpenAI API with parallel processing and chunked caching."""
    # Filter out None values and empty strings
    texts = filter_invalid_texts(texts)
    
    # Setup cache
    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [text for text in texts if text not in text2embedding]
    
    if not texts_to_embed:
        return text2embedding
    
    from .llm_api import get_client
    client = get_client()
    
    # Process in chunks
    next_chunk_idx = _get_next_chunk_index(cache_name)
    
    # Create chunk ranges
    chunk_ranges = [(i, min(i+chunk_size, len(texts_to_embed))) 
                   for i in range(0, len(texts_to_embed), chunk_size)]
    
    # Outer progress bar for chunks
    chunk_iterator = chunk_ranges
    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", total=len(chunk_ranges))
    
    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = {}
        
        # Process chunk in batches with parallel workers
        batches = [chunk_texts[i:i+batch_size] for i in range(0, len(chunk_texts), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(_embed_batch_openai, batch, model, client, timeout=timeout))
            
            # Process results as they complete
            iterator = concurrent.futures.as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(batches), desc=f"Chunk {next_chunk_idx}")
                
            for future in iterator:
                batch_result = future.result()
                batch_idx = futures.index(future)
                batch = batches[batch_idx]
                
                for text, embedding in zip(batch, batch_result):
                    chunk_embeddings[text] = embedding
                    text2embedding[text] = embedding
        
        # Save completed chunk
        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)
    
    return text2embedding

def get_local_embeddings(
    texts: List[str],
    model: str = "nomic-ai/modernbert-embed-base",
    layer_idx = -2, #penultimate layer
    component: str = "mlp", #"hidden_states", "mlp"
    pooling: str = "mean", # "mean", "cls", "max"
    batch_size: int = 128,
    show_progress: bool = True,
    cache_name: Optional[str] = None,
    chunk_size: int = 50000,
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Dict[str, np.ndarray]:
    """Get embeddings using local SentenceTransformer model with chunked caching."""
    # Filter out None values and empty strings
    texts = filter_invalid_texts(texts)

    if cache_name:
        cache_name = f"{cache_name}_{component}_layer{layer_idx}_{pooling}"
    
    # Setup cache
    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [text for text in texts if text not in text2embedding]
    
    if not texts_to_embed:
        return text2embedding
    
    # Load model
    transformer_model = SentenceTransformer(model, device=device)
    print(f"Loaded model {model} to {device}")
    
    # Process in chunks
    next_chunk_idx = _get_next_chunk_index(cache_name)
    
    # Create chunk ranges
    chunk_ranges = [(i, min(i+chunk_size, len(texts_to_embed))) 
                   for i in range(0, len(texts_to_embed), chunk_size)]
    
    # Outer progress bar for chunks
    chunk_iterator = chunk_ranges
    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", total=len(chunk_ranges))
    
    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = {}
        
        # Process chunk in batches
        batch_iterator = range(0, len(chunk_texts), batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc=f"Chunk {next_chunk_idx}")
            
        for i in batch_iterator:
            batch = chunk_texts[i:i+batch_size]
            if "nomic-ai" in model:
                prefixed_batch = ["clustering: " + text for text in batch]
            elif "instructor" in model:
                prefixed_batch = [["Represent the text for classification: ", text] for text in batch]
            else:
                prefixed_batch = batch
            #batch_embs = transformer_model.encode(prefixed_batch, batch_size=batch_size)
            
            batch_embs = extract_intermediate_representations(
                transformer_model, 
                prefixed_batch, 
                layer_idx=layer_idx,
                component=component,
                pooling=pooling,
                batch_size=batch_size
            )

            for text, embedding in zip(batch, batch_embs):
                chunk_embeddings[text] = embedding
                text2embedding[text] = embedding
        
        # Save completed chunk
        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)

    del transformer_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return text2embedding


def extract_intermediate_representations(
    model, 
    texts: List[str], 
    layer_idx: int = -2,
    component: str = "hidden_states",
    pooling: str = "mean",
    batch_size: int = 128
) -> np.ndarray:
    """Extract intermediate layer representations from SentenceTransformer model."""
    
    # Get the underlying transformer model
    if hasattr(model, '_modules'):
        # SentenceTransformer has modules, get the transformer
        transformer = None
        for module in model._modules.values():
            if hasattr(module, 'auto_model'):
                transformer = module.auto_model
                break
        if transformer is None:
            # Fallback: try to get first module that looks like a transformer
            for module in model._modules.values():
                if hasattr(module, 'config') and hasattr(module, 'embeddings'):
                    transformer = module
                    break
    else:
        transformer = model
    
    if transformer is None:
        raise ValueError("Could not find transformer model in SentenceTransformer")
    
    # Tokenize texts
    tokenizer = model.tokenizer
    encoded = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(model.device)
    
    # Store intermediate outputs
    intermediate_outputs = []
    
    def hook_fn(module, input, output):
        if component == "hidden_states":
            # Output is usually the hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]  # First element is usually hidden states
            else:
                hidden_states = output
            intermediate_outputs.append(hidden_states.detach())
        elif component == "mlp":
            # For MLP output, capture the output of feed-forward layer
            if isinstance(output, tuple):
                mlp_output = output[0]
            else:
                mlp_output = output
            intermediate_outputs.append(mlp_output.detach())
    
    # Register hook on the target layer
    hook_handles = []
    
    if hasattr(transformer, 'encoder') and hasattr(transformer.encoder, 'layer'):
        # BERT-like models
        target_layer_idx = layer_idx if layer_idx >= 0 else len(transformer.encoder.layer) + layer_idx
        target_layer = transformer.encoder.layer[target_layer_idx]
        
        if component == "mlp":
            # Hook on the feed-forward network
            if hasattr(target_layer, 'intermediate'):
                handle = target_layer.intermediate.register_forward_hook(hook_fn)
            elif hasattr(target_layer, 'mlp'):
                handle = target_layer.mlp.register_forward_hook(hook_fn)
            elif hasattr(target_layer, 'feed_forward'):
                handle = target_layer.feed_forward.register_forward_hook(hook_fn)
            else:
                # Fallback: hook on the entire layer
                handle = target_layer.register_forward_hook(hook_fn)
        else:  # hidden_states
            # Hook on the entire layer
            handle = target_layer.register_forward_hook(hook_fn)
            
        hook_handles.append(handle)
    
    elif hasattr(transformer, 'layers'):
        # Some models use 'layers' instead of 'encoder.layer'
        target_layer_idx = layer_idx if layer_idx >= 0 else len(transformer.layers) + layer_idx
        target_layer = transformer.layers[target_layer_idx]
        handle = target_layer.register_forward_hook(hook_fn)
        hook_handles.append(handle)
    
    else:
        raise ValueError(f"Unsupported model architecture for layer extraction: {type(transformer)}")
    
    # Forward pass
    with torch.no_grad():
        # Set model to output hidden states if needed
        original_output_hidden_states = getattr(transformer.config, 'output_hidden_states', False)
        original_output_attentions = getattr(transformer.config, 'output_attentions', False)
        
        transformer.config.output_hidden_states = True
        
        try:
            outputs = transformer(**encoded)
            
            # If no hook was triggered, try to extract from model outputs directly
            if not intermediate_outputs and hasattr(outputs, 'hidden_states'):
                if component == "hidden_states" and outputs.hidden_states:
                    target_idx = layer_idx if layer_idx >= 0 else len(outputs.hidden_states) + layer_idx
                    intermediate_outputs.append(outputs.hidden_states[target_idx])        
        finally:
            # Restore original config
            transformer.config.output_hidden_states = original_output_hidden_states
            transformer.config.output_attentions = original_output_attentions
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    
    if not intermediate_outputs:
        raise ValueError(f"No {component} outputs captured from layer {layer_idx}")
    
    # Pool the intermediate representations
    hidden_states = intermediate_outputs[0]  # Shape: [batch_size, seq_len, hidden_dim]
    attention_mask = encoded['attention_mask']
    

    pooled = pool_hidden_states(hidden_states, attention_mask, pooling)
    
    return pooled.cpu().numpy()

def pool_hidden_states(hidden_states, attention_mask, pooling: str = "mean"):
    """Pool token-level hidden states to sentence-level representations."""
    
    if pooling == "cls":
        # Use [CLS] token (first token)
        return hidden_states[:, 0, :]
    
    elif pooling == "mean":
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    elif pooling == "max":
        # Max pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states = hidden_states.masked_fill(mask_expanded == 0, -1e9)
        return torch.max(hidden_states, dim=1)[0]
    
    else:
        raise ValueError(f"Unsupported pooling method: {pooling}")