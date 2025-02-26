import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

# Load embeddings from SVD, CBOW, and Skip-gram
svd_data = torch.load('svd.pt')
cbow_data = torch.load('cbow.pt')
skipgram_data = torch.load('skipgram.pt')

# Extract embeddings and vocab
models = {
    "SVD": svd_data,
    "CBOW": cbow_data,
    "Skip-gram": skipgram_data
}

for model_name, data in models.items():
    data["embeddings"] = data["embeddings"].numpy()

# Function to get most similar words
def get_most_similar_words(word: str, model: str, top_k: int = 5) -> List[Tuple[str, float]]:
    if word not in models[model]['word2idx']:
        return [(f"'{word}' not in vocabulary", 0.0)]
    
    word_vector = models[model]['embeddings'][models[model]['word2idx'][word]].reshape(1, -1)
    similarities = cosine_similarity(word_vector, models[model]['embeddings'])[0]
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    return [(models[model]['idx2word'][idx], similarities[idx]) for idx in top_indices]

# Function to get the best target word from a context
def get_target_word_from_context(context_words: List[str], model: str) -> str:
    valid_contexts = [models[model]['word2idx'][word] for word in context_words if word in models[model]['word2idx']]
    if not valid_contexts:
        return "No valid context words found in vocabulary."
    
    avg_embedding = np.mean(models[model]['embeddings'][valid_contexts], axis=0).reshape(1, -1)
    similarities = cosine_similarity(avg_embedding, models[model]['embeddings'])[0]
    sorted_indices = np.argsort(similarities)[::-1]
    
    for idx in sorted_indices:
        if models[model]['idx2word'][idx] not in context_words:
            return models[model]['idx2word'][idx]
    
    return "No suitable target word found."

# Function to plot T-SNE visualization
def plot_tsne(model: str, top_n: int = 200):
    words = list(models[model]['word2idx'].keys())[:top_n]
    word_indices = [models[model]['word2idx'][word] for word in words]
    embeddings_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(models[model]['embeddings'][word_indices])
    
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
    
    for idx, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]), fontsize=8, alpha=0.7)
    
    plt.title(f"T-SNE Visualization of {model} Embeddings")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.show()

# Example Usage
print(get_most_similar_words("government", "SVD", 5))
print(get_most_similar_words("government", "CBOW", 5))
print(get_most_similar_words("government", "Skip-gram", 5))

print(get_target_word_from_context(["law", "policy", "state"], "SVD"))
print(get_target_word_from_context(["law", "policy", "state"], "CBOW"))
print(get_target_word_from_context(["law", "policy", "state"], "Skip-gram"))

# Plot T-SNE for all models
plot_tsne("SVD")
plot_tsne("CBOW")
plot_tsne("Skip-gram")
