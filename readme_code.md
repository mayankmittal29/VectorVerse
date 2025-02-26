# Word Embeddings Implementation (SVD, CBOW, Skip-gram)

This assignment contains implementations of three word embedding methods: SVD (Singular Value Decomposition), CBOW (Continuous Bag of Words), and Skip-gram. All implementations use the Brown Corpus for training and are evaluated on the WordSim-353 dataset.

## Requirements

```
python >= 3.8
torch >= 1.8.0
numpy >= 1.19.0
nltk >= 3.6.0
scikit-learn >= 0.24.0
scipy >= 1.6.0
tqdm >= 4.60.0
```

Install requirements using:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── README.md
├── requirements.txt
├── svd.py
├── cbow.py
├── skipgram.py
├── wordsim.py
├── wordsim353crowd.csv
├── svd.pt
├── cbow.pt
├── skipgram.pt
├── 2022101094_Report.pdf
```

## Training Word Embeddings

### SVD Implementation
```bash
python3 svd.py
```
This will create `svd.pt` containing the embeddings and vocabulary mappings.

### CBOW Implementation
```bash
python3 cbow.py
```
This will create `cbow.pt` containing the embeddings and vocabulary mappings.

### Skip-gram Implementation
```bash
python3 skipgram.py
```
This will create `skipgram.pt` containing the embeddings and vocabulary mappings.

## Word Similarity Evaluation

To evaluate the embeddings on WordSim-353:
```bash
python3 wordsim.py <embedding_path>.pt
```
Example:
```bash
python3 wordsim.py svd.pt
```

## Loading Saved Embeddings

```python
import torch

# Load embeddings
checkpoint = torch.load('svd.pt')  # or cbow.pt or skipgram.pt

# Access components
embeddings = checkpoint['embeddings']  # Tensor of shape (vocab_size, embedding_dim)
word2idx = checkpoint['word2idx']      # Dictionary mapping words to indices
idx2word = checkpoint['idx2word']      # Dictionary mapping indices to words

# Get embedding for a specific word
word = "example"
if word in word2idx:
    word_idx = word2idx[word]
    word_embedding = embeddings[word_idx]
```

## Model Parameters

All models are trained with the following parameters:
- Embedding dimension: 300
- Context window size: 2
- Minimum word frequency: 5

Additional parameters for CBOW and Skip-gram:
- Batch size: 64-2048
- Number of epochs: 10
- Learning rate: 0.001
- Optimizer: Adam

## File Descriptions

- `svd.py`: Implements SVD-based word embeddings using co-occurrence matrix
- `cbow.py`: Implements CBOW model with negative sampling
- `skipgram.py`: Implements Skip-gram model with negative sampling
- `wordsim.py`: Evaluates embeddings using WordSim-353 dataset
- `requirements.txt`: Lists all Python dependencies
- `2022101094_Report.pdf`: Contains detailed analysis and results

## Results Format

The word similarity evaluation produces a CSV file with the following columns:
- Word 1
- Word 2
- Human Similarity Score
- Model Similarity Score

## Troubleshooting

1. If you encounter CUDA out of memory errors:
   - Reduce batch size
   - Use CPU by setting `device = torch.device('cpu')`

2. If NLTK data is missing:
```python
import nltk
nltk.download('brown')
```

3. For vocabulary-related issues:
   - Check if words are being properly tokenized
   - Verify minimum frequency threshold
   - Ensure proper handling of case sensitivity

