import pickle
import re
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
from utils import plot_word_frequency

def text_split(content: str) -> List[str]:
    """
    Tokenizes the raw text, with several preprocessing cleaning operations.
    :param content: Input text string.
    :return: List of tokens.
    """
    content = re.sub(r"([.!?])", r" \1", content)  # Add a space before .!? characters
    content = re.sub(r"[^a-zA-Z.!?]+", r" ", content)  # Remove all non-alphabetic and non-punctuation characters
    tokens = [word.lower() for word in content.split()]  # Convert to lowercase and split
    return tokens


class Vocabulary:
    UNK_TAG = "<UNK>"  # Unknown token representation
    PAD_TAG = "<PAD>"  # Padding token representation
    UNK = 0  # Numeric representation for UNK token
    PAD = 1  # Numeric representation for PAD token

    def __init__(self):
        self.word2id_dict = {self.UNK_TAG: self.UNK, self.PAD_TAG: self.PAD}
        self.id2word_dict = None
        self.count = {}  # Dictionary to store word frequencies

    def fit(self, sentence: List[str]) -> None:
        """
        Updates word frequency count based on the given sentence.
        :param sentence: List of tokens in the sentence.
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_freq: int = 1, max_freq: int = None, max_vocab_size: int = None) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Builds a vocabulary dictionary based on word frequency criteria.
        :param min_freq: Minimum frequency for words to be included.
        :param max_freq: Maximum frequency for words to be included.
        :param max_vocab_size: Maximum size of the vocabulary.
        :return: Vocabulary dictionary and inverse vocabulary dictionary.
        """
        # Truncate vocabulary based on frequency criteria
        if min_freq > 1:
            self.count = {word: value for word, value in self.count.items() if value >= min_freq}
        if max_freq is not None:
            self.count = {word: value for word, value in self.count.items() if value <= max_freq}
        
        # Limit vocabulary size
        if max_vocab_size is not None:
            raw_len = len(self.count)
            vocab_size = min(max_vocab_size, raw_len)
            print(f'Original vocabulary size: {raw_len}, truncated size: {vocab_size}')
            temp = sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
            self.count = dict(temp)

        # Build vocabulary: token -> index
        for word in self.count:
            self.word2id_dict[word] = len(self.word2id_dict)
        
        # Invert vocabulary: index -> token
        self.id2word_dict = {index: word for word, index in self.word2id_dict.items()}

        return self.word2id_dict, self.id2word_dict

    def __len__(self) -> int:
        return len(self.word2id_dict)


if __name__ == '__main__':
    max_vocab_size = 50257  # GPT2's config
    BASE_DIR = os.path.dirname(__file__)
    data_path = os.path.join(BASE_DIR, '../data')
    out_dir = os.path.join(BASE_DIR, '../outputs')

    # Ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    word2id_dict_path = os.path.join(out_dir, "word2id_dict.npy")
    id2word_dict_path = os.path.join(out_dir, "id2word_dict.npy")
    count_path = os.path.join(out_dir, "word_count.pkl")

    # Initialize vocabulary and count word frequencies
    vocab_hist = Vocabulary()
    if os.path.exists(data_path):
        file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith(".txt")]
        for file_path in tqdm(file_paths, desc=f'Processing files in {data_path}'):
            with open(file_path, encoding='utf-8') as file:
                sentence = text_split(file.read())
                vocab_hist.fit(sentence)

    # Build vocabulary
    word2id_dict, id2word_dict = vocab_hist.build_vocab(min_freq=2, max_vocab_size=(max_vocab_size - 2))  # Exclude UNK and PAD from max size
    
    # Save vocabulary
    np.save(word2id_dict_path, word2id_dict)
    np.save(id2word_dict_path, id2word_dict)
    with open(count_path, 'wb') as f:
        pickle.dump(vocab_hist.count, f)

    # Visualize word frequency
    print(f'Vocabulary size: {len(word2id_dict)}')
    plot_word_frequency(vocab_hist.count)