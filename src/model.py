import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, hidden):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden

        # Define two embedding layers: one for input words, one for context words
        self.in_embedding = nn.Embedding(self.vocab_size, self.hidden)
        self.out_embedding = nn.Embedding(self.vocab_size, self.hidden)
    
    def forward(self, input_labels, pos_labels, neg_labels):
        # Embed the center word [batch, hidden]
        # Embed the positive (context) words [batch, window * 2, hidden]
        # Embed the negative samples [batch, window * 2 * k, hidden]
        input_embedding = self.in_embedding(input_labels)
        pos_embedding = self.out_embedding(pos_labels)
        neg_embedding = self.out_embedding(neg_labels)

        # Add a dimension for batch matrix multiplication [batch, hidden, 1]
        input_embedding = input_embedding.unsqueeze(2)

        # Calculate the similarity
        esp = 1e-9 # To avoid instability
        # Positive samples and the center word [batch, window * 2, 1]
        # Negative samples and the center word [batch, window * 2 * k, 1]
        pos_dot = torch.bmm(pos_embedding, input_embedding) + esp
        neg_dot = torch.bmm(neg_embedding, -input_embedding) + esp

        # Remove the last dimension [batch, window * 2] and [batch, window * 2 * k]
        pos_dot = pos_dot.squeeze(2)
        neg_dot = neg_dot.squeeze(2)

        # Calculate positive sample loss
        # Calculate negative sample loss
        pos_loss = F.logsigmoid(pos_dot).sum(1)
        neg_loss = F.logsigmoid(neg_dot).sum(1)

        # Total loss
        loss = neg_loss + pos_loss

        return -loss
    
    def get_input_embedding(self):
        # Get the input embedding weights for evaluation
        return self.in_embedding.weight.detach()