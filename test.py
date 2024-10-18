import numpy as np
from scipy.spatial.distance import cosine
import torch

# Load vocabulary mapping
with open('./outputs/word2id_dict.npy', 'rb') as f:
    word2idx = np.load(f, allow_pickle=True).item()

# Load embedding weights

# This mehtod should have been removed but I keep it
# Cuz I cannot train the model again due to the high cost of computing resource
embedding_weights = torch.load('./outputs/model.pth', map_location=torch.device('cpu'), weights_only=True)['in_embedding.weight'].detach().numpy()
# # The revised method is this one
# embedding_weights = torch.load('./outputs/embedding_weights.pth', map_location=torch.device('cpu')).numpy()

# Get idx2word mapping
idx2word = {idx: word for word, idx in word2idx.items()}

def find_nearest(word, embedding_weights, word2idx, idx2word):
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary.")
        return []

    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([cosine(embedding, e) for e in embedding_weights])
    nearest_indices = cos_dis.argsort()[1:11]  # Exclude the word itself (distance=0)
    return [idx2word[i] for i in nearest_indices]

# Test similarity search
test_words = ['two', 'man', 'computers', 'machine']
for word in test_words:
    similar_words = find_nearest(word, embedding_weights, word2idx, idx2word)
    print(f"Word: '{word}' is similar to: {similar_words}")

# Save results
results_path = './outputs/similar_words_results.txt'
with open(results_path, 'w') as f:
    for word in test_words:
        similar_words = find_nearest(word, embedding_weights, word2idx, idx2word)
        f.write(f"Word: '{word}' is similar to: {similar_words}\n")