#!/usr/bin/env python3
"""Test script for profiling annotation with local LLM on Yelp data."""

import random
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from hypothesaes.annotate import annotate_texts_with_concepts
from hypothesaes.utils import get_text_for_printing

def main():
    parser = argparse.ArgumentParser(description="Profile annotation with local LLM on Yelp data")
    parser.add_argument("--n_rows", type=int, default=1000, help="Number of rows to process")
    parser.add_argument("--concept", type=str, default="mentions anticipation or plans to return to the restaurant or order again", 
                       help="Concept to annotate")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", 
                       help="LLM to use for annotation")
    args = parser.parse_args()

    # Load demo data
    print("Loading Yelp demo data...")
    train_df = pd.read_json("../demo-data/yelp-demo-train-20K.json", lines=True)

    # Take specified number of rows
    df_subset = train_df.head(args.n_rows)
    texts = df_subset['text'].tolist()
    stars = df_subset['stars'].values

    # Annotate with local LLM
    print(f"Annotating {len(texts)} texts with concept: '{args.concept}'")
    print(f"Using LLM: {args.model}")

    start_time = time.time()
    annotations = annotate_texts_with_concepts(
        texts=texts,
        concepts=[args.concept],
        model=args.model,
        n_workers=100, # has no effect for local LLM
        show_progress=True
    )
    end_time = time.time()

    # Extract annotation array
    concept_annotations = annotations[args.concept]

    # Print timing
    duration = end_time - start_time
    print(f"\nAnnotation completed in {duration:.2f} seconds")

    # Print examples
    print("\n=== Examples annotated as 1 ===")
    positive_indices = np.where(concept_annotations == 1)[0]
    sample_size = min(5, len(positive_indices))
    selected_indices = random.sample(positive_indices.tolist(), sample_size)
    for i in selected_indices:
        print(f"{i+1}. {get_text_for_printing(texts[i])}...")

    print("\n=== Examples annotated as 0 ===")
    negative_indices = np.where(concept_annotations == 0)[0]
    sample_size = min(5, len(negative_indices))
    selected_indices = random.sample(negative_indices.tolist(), sample_size)
    for i in selected_indices:
        print(f"{i+1}. {get_text_for_printing(texts[i])}...")

    # Calculate correlation with stars
    correlation, p_value = pearsonr(concept_annotations, stars)
    print(f"\n=== Correlation Analysis ===")
    print(f"Pearson correlation between annotations and stars: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Total annotations: {len(concept_annotations)}")
    print(f"Positive annotations: {np.sum(concept_annotations == 1)}")
    print(f"Negative annotations: {np.sum(concept_annotations == 0)}")

if __name__ == "__main__":
    main()
