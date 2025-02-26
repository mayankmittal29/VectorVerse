# import torch
# import numpy as np
# from typing import Dict, List
# from tqdm import tqdm
# import re
# from collections import defaultdict, Counter
# from nltk.corpus import brown
# import nltk
# from torch import nn
# import torch.optim as optim

# # Download the Brown corpus if not already present
# nltk.download('brown')

# # Parameters
# window_size = 2  # Context window size
# embedding_dim = 100  # Number of dimensions for word embeddings
# min_freq = 5  # Minimum frequency threshold
# batch_size = 64
# num_epochs = 10
# learning_rate = 0.001

# # # Stop words to exclude from analysis
# STOP_WORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
#                   "yourself", "he", "him", "his", "himself", "she", "her", "hers", "it", "its", "itself", 
#                   "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", 
#                   "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", 
#                   "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
#                   "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", 
#                   "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", 
#                   "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"])

# class SkipgramModel(nn.Module):
#     def __init__(self, vocab_size: int, embedding_dim: int):
#         super(SkipgramModel, self).__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.linear = nn.Linear(embedding_dim, vocab_size)
        
#     def forward(self, inputs):
#         embeds = self.embeddings(inputs)
#         output = self.linear(embeds)
#         return output

# def create_skipgram_dataset(tokens, word2idx, window_size):
#     data = []
#     for i in range(len(tokens)):
#         for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
#             if i != j and tokens[i] in word2idx and tokens[j] in word2idx:
#                 input_idx = torch.tensor(word2idx[tokens[i]])
#                 target_idx = torch.tensor(word2idx[tokens[j]])
#                 data.append((input_idx, target_idx))
#     return data

# def main():
#     # Load and preprocess Brown corpus
#     tokens = [word.lower() for word in brown.words() if word.isalpha() and word.lower() not in STOP_WORDS]
#     word_freq = Counter(tokens)
    
#     # Build vocabulary
#     word2idx = {}
#     idx2word = {}
#     idx = 0
#     for word, freq in tqdm(word_freq.items(), desc="Building vocabulary"):
#         if freq >= min_freq:
#             word2idx[word] = idx
#             idx2word[idx] = word
#             idx += 1
            
#     vocab_size = len(word2idx)
#     print(f"Vocabulary size: {vocab_size}")
    
#     # Create dataset
#     dataset = create_skipgram_dataset(tokens, word2idx, window_size)
    
#     # Initialize model and training parameters
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = SkipgramModel(vocab_size, embedding_dim).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     for epoch in range(num_epochs):
#         total_loss = 0
#         model.train()
        
#         for i in tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
#             batch = dataset[i:i+batch_size]
#             input_batch = torch.stack([item[0] for item in batch]).to(device)
#             target_batch = torch.stack([item[1] for item in batch]).to(device)
            
#             optimizer.zero_grad()
#             output = model(input_batch)
#             loss = criterion(output, target_batch)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#         avg_loss = total_loss / (len(dataset) // batch_size)
#         print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
#     # Save embeddings and vocabulary
#     embeddings = model.embeddings.weight.data.cpu()
#     torch.save({
#         'embeddings': embeddings,
#         'word2idx': word2idx,
#         'idx2word': idx2word
#     }, 'skipgram.pt')
#     print("Model saved successfully!")

# if __name__ == "__main__":
#     main()

#skipgram.py
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.corpus import brown
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Download Brown corpus if not already present
nltk.download('brown')

# Parameters
WINDOW_SIZE = 2
EMBEDDING_DIM = 300
MIN_FREQ = 5
BATCH_SIZE = 2048
NEGATIVE_SAMPLES = 5
EPOCHS = 10
LEARNING_RATE = 1e-3

STOP_WORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"])


class SkipgramDataset(Dataset):
    def __init__(self, tokens: List[str], word2idx: Dict[str, int], window_size: int, 
                 negative_sampling_table: torch.Tensor):
        self.tokens = tokens
        self.word2idx = word2idx
        self.window_size = window_size
        self.negative_sampling_table = negative_sampling_table
        
        # Pre-compute all center-context pairs
        self.pairs = []
        for i in range(window_size, len(tokens) - window_size):
            target_word = tokens[i]
            # Get context words from both sides of the target
            for j in range(-window_size, window_size + 1):
                if j != 0:  # Skip the target word itself
                    context_pos = i + j
                    if 0 <= context_pos < len(tokens):
                        self.pairs.append((target_word, tokens[context_pos]))
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        target_word, context_word = self.pairs[idx]
        
        # Convert words to indices
        target_idx = torch.tensor(self.word2idx[target_word], dtype=torch.long)
        context_idx = torch.tensor(self.word2idx[context_word], dtype=torch.long)
        
        # Generate negative samples
        negative_samples = self.negative_sampling_table[torch.randint(
            len(self.negative_sampling_table), (NEGATIVE_SAMPLES,))]
        
        return target_idx, context_idx, negative_samples


class Skipgram:
    def __init__(self, vocab_size: int, embedding_dim: int):
        # Initialize target and context embedding tables with uniform distribution
        self.target_embeddings = torch.empty(vocab_size, embedding_dim)
        self.context_embeddings = torch.empty(vocab_size, embedding_dim)
        
        # Initialize with small random values
        self.target_embeddings.uniform_(-0.1, 0.1)
        self.context_embeddings.uniform_(-0.1, 0.1)
        
        # Enable gradient computation
        self.target_embeddings.requires_grad = True
        self.context_embeddings.requires_grad = True
        
    def to(self, device):
        self.target_embeddings = self.target_embeddings.to(device)
        self.context_embeddings = self.context_embeddings.to(device)
        return self
        
    def parameters(self):
        return [self.target_embeddings, self.context_embeddings]
        
    def train(self):
        self.training = True
        
    def forward(self, target_words, context_words, negative_samples):
        # Manual embedding lookup
        target_embeds = self.target_embeddings[target_words]  # [batch_size, embed_dim]
        context_embeds = self.context_embeddings[context_words]  # [batch_size, embed_dim]
        
        # Handle negative samples
        negative_embeds = self.context_embeddings[negative_samples.view(-1)]
        negative_embeds = negative_embeds.view(negative_samples.size(0), negative_samples.size(1), -1)
        
        # Compute positive scores
        positive_scores = torch.sum(target_embeds * context_embeds, dim=1)  # [batch_size]
        
        # Compute negative scores
        negative_scores = torch.bmm(negative_embeds, 
                                  target_embeds.unsqueeze(2)).squeeze()  # [batch_size, neg_samples]
        
        return positive_scores, negative_scores


def create_negative_sampling_table(word_freq: Counter, word2idx: Dict[str, int], table_size: int = 1000000):
    vocab_size = len(word2idx)
    freq_array = np.zeros(vocab_size)
    for word, idx in word2idx.items():
        freq_array[idx] = word_freq[word]
    
    # Apply power of 0.75 to frequencies
    adjusted_freq = freq_array ** 0.75
    prob_dist = adjusted_freq / adjusted_freq.sum()
    
    # Create table
    table = np.random.choice(vocab_size, size=int(table_size), p=prob_dist)
    return torch.LongTensor(table)


def clip_grad_norm(parameters, max_norm):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for target_words, context_words, negative_samples in tqdm(train_loader, desc="Training"):
        target_words = target_words.to(device)
        context_words = context_words.to(device)
        negative_samples = negative_samples.to(device)
        
        # Forward pass
        positive_scores, negative_scores = model.forward(target_words, context_words, negative_samples)
        
        # Compute loss
        positive_loss = torch.mean(torch.log(torch.sigmoid(positive_scores) + 1e-10))
        negative_loss = torch.mean(torch.sum(torch.log(torch.sigmoid(-negative_scores) + 1e-10), dim=1))
        loss = -(positive_loss + negative_loss)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients manually
        clip_grad_norm(model.parameters(), 5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


# Preprocess data
print("Loading and preprocessing data...")
tokens = [word.lower() for word in brown.words() 
         if word.isalpha() and word.lower() not in STOP_WORDS]
word_freq = Counter(tokens)

# Create vocabulary
word2idx = {}
idx2word = {}
idx = 0
for word, freq in word_freq.items():
    if freq >= MIN_FREQ:
        word2idx[word] = idx
        idx2word[idx] = word
        idx += 1

vocab_size = len(word2idx)
print(f"Vocabulary size: {vocab_size}")

# Filter tokens
filtered_tokens = [word for word in tokens if word in word2idx]
print(f"Total tokens after filtering: {len(filtered_tokens)}")

# Create negative sampling table
negative_sampling_table = create_negative_sampling_table(word_freq, word2idx)

# Create dataset and dataloader
dataset = SkipgramDataset(filtered_tokens, word2idx, WINDOW_SIZE, negative_sampling_table)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Skipgram(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("Starting training...")
losses = []
try:
    for epoch in range(EPOCHS):
        loss = train_model(model, dataloader, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Skip-gram Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Get final embeddings (using target embeddings as final embeddings)
    embeddings = model.target_embeddings.detach().cpu()
    embeddings_numpy = normalize(embeddings.numpy())

    # Save embeddings and vocabulary
    print(f"Saving embeddings...")
    torch.save({
        'embeddings': torch.tensor(embeddings_numpy, dtype=torch.float32),
        'word2idx': word2idx,
        'idx2word': idx2word
    }, 'skipgram_embeddings.pt')
    print("Embeddings saved successfully!")

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current embeddings...")
    embeddings = model.target_embeddings.detach().cpu()
    embeddings_numpy = normalize(embeddings.numpy())
    torch.save({
        'embeddings': torch.tensor(embeddings_numpy, dtype=torch.float32),
        'word2idx': word2idx,
        'idx2word': idx2word
    }, 'skipgram_embeddings_interrupted.pt')
    print("Partial embeddings saved successfully!")