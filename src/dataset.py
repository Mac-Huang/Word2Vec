import torch
import torch.utils.data as tud
from typing import List, Dict
from build_vocab import Vocabulary

class EmbeddingDataset(tud.Dataset):
    def __init__(self, text: List[str], word2idx: Dict[str, int], word_freqs: List[float], window_size: int, negative_sample: int):
        super(EmbeddingDataset, self).__init__()
        # Adds a new word to the vocabulary if there is space available.
        # It could be more dynamic.
        self.text_encoded = [word2idx.get(word, Vocabulary.add_word(word)) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        word_freqs_list = [word_freqs.get(word, 0.0) for word in word2idx.keys()]
        self.word_freqs = torch.Tensor(word_freqs_list)
        self.window_size = window_size
        self.negative_sample = negative_sample

    def __len__(self) -> int:
        return len(self.text_encoded)

    def __getitem__(self, idx: int):
        center_word = self.text_encoded[idx]
        # Get words in the window, excluding the center word
        pos_idx = [i for i in range(idx - self.window_size, idx)] + [i for i in range(idx + 1, idx + self.window_size + 1)]
        pos_idx = [i % len(self.text_encoded) for i in pos_idx]

        pos_words = self.text_encoded[pos_idx]

        neg_mask = self.word_freqs.clone()
        neg_mask[pos_words] = 0

        neg_words = torch.multinomial(neg_mask, self.negative_sample * pos_words.shape[0], True)
        # Check if negative sample failure exists
        if len(set(pos_words.numpy().tolist()) & set(neg_words.numpy().tolist())) > 0:
            print('Need to resample.')

        return center_word, pos_words, neg_words