# #svd.py
# import torch
# from typing import Dict, List
# from tqdm import tqdm
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict, Counter
# from nltk.corpus import brown
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.preprocessing import normalize
# import nltk

# # Download the Brown corpus if not already present
# nltk.download('brown')

# # Parameters
# window_size = 2  # Context window size
# embedding_dim = 500  # Number of dimensions for word embeddings
# min_freq = 2  # Minimum frequency threshold to include a word in the vocab

# # Stop words to exclude from analysis
# STOP_WORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"])

# # Load and preprocess Brown corpus
# tokens = [word.lower() for word in brown.words() if word.isalpha() and word.lower() not in STOP_WORDS]

# word_freq = Counter(tokens)

# # Filter words based on min_freq
# vocab = {word for word, freq in word_freq.items() if freq >= min_freq}

# # Build co-occurrence matrix
# co_occurrence = defaultdict(lambda: defaultdict(int))
# filtered_tokens = [word for word in tokens if word in vocab]

# for i, word in enumerate(filtered_tokens):
#     for j in range(max(0, i - window_size), min(len(filtered_tokens), i + window_size + 1)):
#         if i != j:
#             co_occurrence[word][filtered_tokens[j]] += 1
            

# # Convert co-occurrence dictionary to a matrix
# co_occurrence_keys = list(co_occurrence.keys())
# word_to_id = {word: i for i, word in enumerate(co_occurrence_keys)}

# matrix = np.zeros((len(co_occurrence_keys), len(co_occurrence_keys)))
# for word1, coocs in co_occurrence.items():
#     for word2, count in coocs.items():
#         matrix[word_to_id[word1]][word_to_id[word2]] = count
        
# # Apply Truncated SVD for word embeddings
# svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
# embeddings = svd.fit_transform(matrix)
# embeddings = normalize(embeddings)  # Normalize embeddings

# # Perform PCA for visualization
# pca = PCA(n_components=2)
# embeddings_2d = pca.fit_transform(embeddings)

# # Select top-N frequent words for plotting
# top_n = 200
# top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:top_n]
# top_indices = [word_to_id[word] for word in top_words if word in word_to_id]

# plt.figure(figsize=(12, 8))
# plt.scatter(embeddings_2d[top_indices, 0], embeddings_2d[top_indices, 1], alpha=0.5)

# for idx, word in zip(top_indices, top_words):
#     x, y = embeddings_2d[idx]
#     plt.annotate(word, (x, y), fontsize=8, alpha=0.7)

# plt.title(f"2D PCA of Top {top_n} Word SVD Embeddings (Dim={embedding_dim}, Window={window_size})")
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.tight_layout()
# plt.show()

# word2idx = {}
# idx2word = {}
        
# print(f"Found {len(word_freq)} unique words")
# # Build vocabulary with words above min_count
# idx = 0
# for word, freq in tqdm(word_freq.items(), desc="Filtering vocabulary"):
#             if freq >= min_freq:
#                 word2idx[word] = idx
#                 idx2word[idx] = word
#                 idx += 1
                
# # Save embeddings as a .pt file
# embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
# print(f"\nSaving embeddings to {'svd4.pt'}...")
# # Save both embeddings and vocabulary
# torch.save({
#             'embeddings': embeddings_tensor,
#             'word2idx': word2idx,
#             'idx2word': idx2word
#     }, 'svd4.pt')
# print("Embeddings saved successfully!")

# import torch
# from typing import Dict, List
# from tqdm import tqdm
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict, Counter
# from nltk.corpus import brown
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import normalize
# import nltk

# # Download the Brown corpus if not already present
# nltk.download('brown')

# # Parameters
# window_size = 2  # Context window size
# embedding_dim = 100  # Number of dimensions for word embeddings
# min_freq = 5  # Minimum frequency threshold to include a word in the vocab

# # Stop words to exclude from analysis
# STOP_WORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"])

# # Load and preprocess Brown corpus
# tokens = [word.lower() for word in brown.words() if word.isalpha() and word.lower() not in STOP_WORDS]

# word_freq = Counter(tokens)

# # Filter words based on min_freq
# vocab = {word for word, freq in word_freq.items() if freq >= min_freq}

# # Build co-occurrence matrix
# co_occurrence = defaultdict(lambda: defaultdict(int))
# filtered_tokens = [word for word in tokens if word in vocab]

# for i, word in enumerate(filtered_tokens):
#     for j in range(max(0, i - window_size), min(len(filtered_tokens), i + window_size + 1)):
#         if i != j:
#             co_occurrence[word][filtered_tokens[j]] += 1

# # Convert co-occurrence dictionary to a matrix
# co_occurrence_keys = list(co_occurrence.keys())
# word_to_id = {word: i for i, word in enumerate(co_occurrence_keys)}

# matrix = np.zeros((len(co_occurrence_keys), len(co_occurrence_keys)))
# for word1, coocs in co_occurrence.items():
#     for word2, count in coocs.items():
#         matrix[word_to_id[word1]][word_to_id[word2]] = count

# # Perform Full SVD (U, Sigma, V^T)
# U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
# embeddings = U[:, :embedding_dim] * S[:embedding_dim]  # Compress embeddings
# embeddings = normalize(embeddings)  # Normalize embeddings

# # Perform PCA for visualization
# pca = PCA(n_components=2)
# embeddings_2d = pca.fit_transform(embeddings)

# # Select top-N frequent words for plotting
# top_n = 200
# top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:top_n]
# top_indices = [word_to_id[word] for word in top_words if word in word_to_id]

# plt.figure(figsize=(12, 8))
# plt.scatter(embeddings_2d[top_indices, 0], embeddings_2d[top_indices, 1], alpha=0.5)

# for idx, word in zip(top_indices, top_words):
#     x, y = embeddings_2d[idx]
#     plt.annotate(word, (x, y), fontsize=8, alpha=0.7)

# plt.title(f"2D PCA of Top {top_n} Word SVD Embeddings (Dim={embedding_dim}, Window={window_size})")
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.tight_layout()
# plt.show()

# word2idx = {}
# idx2word = {}

# print(f"Found {len(word_freq)} unique words")
# # Build vocabulary with words above min_count
# idx = 0
# for word, freq in tqdm(word_freq.items(), desc="Filtering vocabulary"):
#     if freq >= min_freq:
#         word2idx[word] = idx
#         idx2word[idx] = word
#         idx += 1

# # Save embeddings as a .pt file
# embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
# print(f"\nSaving embeddings to {'full_svd.pt'}...")
# # Save both embeddings and vocabulary
# torch.save({
#     'embeddings': embeddings_tensor,
#     'word2idx': word2idx,
#     'idx2word': idx2word
# }, 'full_svd.pt')
# print("Embeddings saved successfully!")


import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from nltk.corpus import brown
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as sp_linalg
import nltk

# Download the Brown corpus if not already present
nltk.download('brown')

# Parameters
window_size = 2  # Context window size
embedding_dim = 300  # Number of dimensions for word embeddings
min_freq = 5  # Minimum frequency threshold to include a word in the vocab

# Stop words to exclude from analysis
STOP_WORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"])

# Load and preprocess Brown corpus
tokens = [word.lower() for word in brown.words() if word.isalpha() and word.lower() not in STOP_WORDS]


# Count word frequencies
word_freq = Counter(tokens)

# Build vocabulary (includes all words)
vocab = list({word for word, freq in word_freq.items() if freq >= min_freq})
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

# Build sparse co-occurrence matrix
row, col, data = [], [], []
for i in tqdm(range(len(tokens)), desc="Building Co-occurrence Matrix"):
    word = tokens[i]
    if word not in word_to_id:  # Skip words not in vocab
        continue
    word_id = word_to_id[word]

    for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
        if i != j:
            context_word = tokens[j]
            if context_word not in word_to_id:  # Skip if context word is not in vocab
                continue
            context_id = word_to_id[context_word]

            row.append(word_id)
            col.append(context_id)
            data.append(1)  # Increment count


# Convert to sparse matrix
vocab_size = len(vocab)
co_occurrence_matrix = csr_matrix((data, (row, col)), shape=(vocab_size, vocab_size))

# Compute full SVD
print("\nPerforming Full SVD on Sparse Matrix...")
U, S, Vt = sp_linalg.svds(co_occurrence_matrix, k=embedding_dim)

# Normalize embeddings
embeddings = normalize(U @ np.diag(S))

# Save embeddings as a .pt file
embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
print("\nSaving embeddings to 'full_svd.pt'...")

torch.save({
    'embeddings': embeddings_tensor,
    'word2idx': word_to_id,
    'idx2word': id_to_word
}, 'full_svd.pt')

print("Embeddings saved successfully!")
