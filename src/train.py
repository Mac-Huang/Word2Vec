import torch
from torch.utils.data import DataLoader
from skip_gram import Word2Vec
from dataset import EmbeddingDataset
import torch.optim as optim
from tqdm import tqdm
from time import time
import pickle
import numpy as np
from utils import save_training_results

# Define training parameters
VOCAB_SIZE = 50257  # GPT2's config
EMBEDDING_DIM = 128  # Embedding dimension from origin papper
BATCH_SIZE = 8192  # Batch size
EPOCHS = 5  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate
WINDOW_SIZE = 5  # Context window size
NEGATIVE_SAMPLES = 15  # Number of negative samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load dataset
with open('../outputs/word2id_dict.npy', 'rb') as f:
    word2idx = np.load(f, allow_pickle=True).item()
with open('../outputs/id2word_dict.npy', 'rb') as f:
    id2word = np.load(f, allow_pickle=True).item()
with open('../outputs/word_count.pkl', 'rb') as f:
    word_freqs = pickle.load(f)
with open('../data/text8.txt', 'r') as f:
    text = f.read().split()

# Create dataset and dataloader
dataset = EmbeddingDataset(text, word2idx, word_freqs, WINDOW_SIZE, NEGATIVE_SAMPLES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = Word2Vec(VOCAB_SIZE, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
loss_history = []
batch_loss_history = []
training_start_time = time()

for epoch in range(EPOCHS):
    total_loss = 0
    start_time = time()
    for step, (input_labels, pos_labels, neg_labels) in enumerate(tqdm(dataloader)):
        # Move data to GPU if available
        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)

        # Training step: zero gradients, calculate loss, backpropagate
        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_loss_history.append(loss.item())

        # Print loss every 1000 steps
        if step % 100 == 0 and step != 0:
            end_time = time()
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Time: {end_time - start_time:.2f}s")
            start_time = time()

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch}, Average Loss: {avg_loss}")

# Save model and vocabulary mapping
torch.save(model.state_dict(), '../outputs/model.pth')
embedding_weights = model.get_input_embedding().cpu()

# Save results using utils
training_end_time = time()
training_duration = training_end_time - training_start_time
hyperparameters = {
    "VOCAB_SIZE": VOCAB_SIZE,
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LEARNING_RATE": LEARNING_RATE,
    "WINDOW_SIZE": WINDOW_SIZE,
    "NEGATIVE_SAMPLES": NEGATIVE_SAMPLES,
    "TRAINING_DURATION": training_duration
}

save_training_results(loss_history, batch_loss_history, hyperparameters, '../outputs')