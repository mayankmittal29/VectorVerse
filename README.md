# 🔤 VectorVerse: Static Word Embeddings Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A comprehensive implementation of static word embedding techniques for natural language processing, featuring SVD, CBOW, and Skip-gram models.

## 📋 Overview

This repository contains implementations of three prominent static word embedding techniques:

1. **SVD (Singular Value Decomposition)** - A frequency-based approach using co-occurrence matrices
2. **CBOW (Continuous Bag of Words)** - A prediction-based neural embedding model
3. **Skip-gram** - A prediction-based neural embedding model with superior performance on semantic tasks

All models are implemented from scratch in PyTorch and trained on the Brown Corpus. The embeddings are evaluated using the WordSim-353 dataset to measure semantic similarity performance.

## 🛠️ Implementation Details

### Data Preprocessing
- Stop word removal
- Non-alphabetic token filtering
- Word frequency thresholding (min freq = 5)
- Context window definition (size = 2)
- Embedding dimension of 300

### Model Architectures
- **SVD**: Co-occurrence matrix + SVD + Normalization
- **CBOW**: Predicts target word from context with negative sampling
- **Skip-gram**: Predicts context words from target with negative sampling

## 📊 Results

Performance on WordSim-353 dataset (Spearman Correlation):

| Model | Spearman Correlation |
|-------|---------------------|
| SVD | 0.17186670 |
| CBOW | 0.29502401 |
| Skip-gram | 0.32181557 |

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training Embeddings

```bash
# Train SVD embeddings
python svd.py

# Train CBOW embeddings
python cbow.py

# Train Skip-gram embeddings
python skipgram.py
```

### Evaluating Word Similarity

```bash
# Evaluate SVD embeddings
python wordsim.py svd.pt

# Evaluate CBOW embeddings
python wordsim.py cbow.pt

# Evaluate Skip-gram embeddings
python wordsim.py skipgram.pt
```

## 📁 Repository Structure

```
├── svd.py             # SVD implementation
├── cbow.py            # CBOW implementation
├── skipgram.py        # Skip-gram implementation
├── wordsim.py         # Word similarity evaluation
├── utils.py           # Utility functions
├── requirements.txt   # Dependencies
├── svd.pt             # Trained SVD embeddings
├── cbow.pt            # Trained CBOW embeddings
├── skipgram.pt        # Trained Skip-gram embeddings
└── report.pdf         # Detailed analysis report
```

## 📊 Visualizations

The repository includes t-SNE visualizations of word embeddings, demonstrating the clustering and relationships captured by each model.

## 🔍 Analysis Highlights

- Skip-gram performs best at capturing semantic relationships
- CBOW offers a good balance between performance and training efficiency
- SVD provides fast training but with limited semantic capture
- Neural models (CBOW and Skip-gram) significantly outperform matrix factorization (SVD)

## 📚 References

1. Mikolov et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
2. Mikolov et al. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
3. Goldberg & Levy (2014). [word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method](https://arxiv.org/abs/1402.3722)
