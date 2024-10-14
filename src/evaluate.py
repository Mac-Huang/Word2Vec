import numpy as np
from scipy.spatial.distance import cosine
import pickle
import torch

# Load vocabulary mapping
with open('./outputs/word2id_dict.pkl', 'rb') as f:
    word2idx = pickle.load(f)

# Load embedding weights
embedding_weights = torch.load('./outputs/model.pth')['in_embedding.weight'].cpu().detach().numpy()

# Get idx2word mapping
idx2word = {idx: word for word, idx in word2idx.items()}

def find_nearest(word, embedding_weights, word2idx, idx2word):
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary.")
        return []

    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([cosine(embedding, e) for e in embedding_weights])
    nearest_indices = cos_dis.argsort()[:10]
    return [idx2word[i] for i in nearest_indices]

# Test similarity search
test_words = ['two', 'man', 'computers', 'machine']
for word in test_words:
    similar_words = find_nearest(word, embedding_weights, word2idx, idx2word)
    print(f"Word: '{word}' is similar to: {similar_words}")