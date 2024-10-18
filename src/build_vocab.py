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

    def build_vocab(self, min_freq: int = 1, max_freq: int = None, max_vocab_size: int = None, buffer_size: int = 1000) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Builds a vocabulary dictionary based on word frequency criteria.
        :param min_freq: Minimum frequency for words to be included.
        :param max_freq: Maximum frequency for words to be included.
        :param max_vocab_size: Maximum size of the vocabulary.
        :param buffer_size: Number of reserved slots for future words.
        :return: Vocabulary dictionary and inverse vocabulary dictionary.
        """
        # Truncate vocabulary based on frequency criteria
        if min_freq > 1:
            self.count = {word: value for word, value in self.count.items() if value >= min_freq}
        if max_freq is not None:
            self.count = {word: value for word, value in self.count.items() if value <= max_freq}

        # Limit vocabulary size, keeping buffer for future additions
        if max_vocab_size is not None:
            raw_len = len(self.count)
            vocab_size = min(max_vocab_size - buffer_size, raw_len)
            print(f'Original vocabulary size: {raw_len}, truncated size: {vocab_size} with buffer: {buffer_size}')
            temp = sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
            self.count = dict(temp)

        # Build vocabulary: token -> index
        for word in self.count:
            self.word2id_dict[word] = len(self.word2id_dict)
            
        # Invert vocabulary: index -> token
        self.id2word_dict = {index: word for word, index in self.word2id_dict.items()}

        return self.word2id_dict, self.id2word_dict
    
    @staticmethod
    def add_word(self, word: str) -> int:
        """
        Adds a new word to the vocabulary if there is space available.
        :param word: The word to add.
        :return: The index of the newly added word or existing word.
        """
        if word not in self.word2id_dict:
            if len(self.word2id_dict) < max_vocab_size:
                new_index = len(self.word2id_dict)
                self.word2id_dict[word] = new_index
                self.id2word_dict[new_index] = word
                self.count[word] = 1  # Initialize the word frequency to 1
                print(f"Added new word '{word}' to vocabulary with index {new_index}")
                return new_index
            else:
                print(f"No space available to add new word '{word}'")
                return self.word2id_dict[self.UNK_TAG]  # Return UNK index if no space is available
        return self.word2id_dict[word]


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