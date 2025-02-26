import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Tuple
import argparse

def load_embeddings(path: str) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load embeddings and vocabulary from saved file."""
    data = torch.load(path)
    return data['embeddings'], data['word2idx']

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return float(torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))

def evaluate_similarity(embeddings: torch.Tensor, 
                       word2idx: Dict[str, int], 
                       word_pairs: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate word similarities using the provided embeddings and WordSim-353 dataset.
    Returns Spearman correlation and detailed results DataFrame.
    """
    results = []
    for _, row in word_pairs.iterrows():
        word1, word2 = row['Word 1'].lower(), row['Word 2'].lower()
        human_score = row['Human (Mean)']
        
        # Skip if either word is not in vocabulary
        if word1 not in word2idx or word2 not in word2idx:
            continue
        
        # Get word vectors
        vec1 = embeddings[word2idx[word1]]
        vec2 = embeddings[word2idx[word2]]
        
        # Compute cosine similarity
        cos_sim = cosine_similarity(vec1, vec2)
        
        results.append({
            'Word 1': word1,
            'Word 2': word2,
            'Human (Mean)': human_score,
            'cosine_similarity': cos_sim
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate Spearman correlation
    correlation, _ = spearmanr(
        results_df['Human (Mean)'],
        results_df['cosine_similarity']
    )
    
    return correlation, results_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate word embeddings on WordSim-353')
    parser.add_argument('embedding_path', type=str, help='Path to the embedding file (.pt)')
    parser.add_argument('--wordsim_path', type=str, default='wordsim353crowd.csv',
                       help='Path to WordSim-353 dataset')
    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embedding_path}")
    embeddings, word2idx = load_embeddings(args.embedding_path)

    # Load WordSim-353 dataset
    print(f"Loading WordSim-353 from {args.wordsim_path}")
    wordsim_df = pd.read_csv(args.wordsim_path)

    # Evaluate similarities
    print("Computing similarities...")
    correlation, results_df = evaluate_similarity(embeddings, word2idx, wordsim_df)
    model_name = args.embedding_path.replace('.pt', '')  # Example: svd.pt → svd
    # Save results
    results_df.to_csv(f'{model_name}.csv', index=False)
    print(f"\nResults saved to {model_name}.csv")
    print(f"Spearman Correlation: {correlation:.4f}")

if __name__ == "__main__":
    main()