import matplotlib.pyplot as plt
from typing import Dict
import matplotlib.pyplot as plt
import json


def plot_word_frequency(word_count_dict: Dict[str, int], hist_size: int = 100) -> None:
    """
    Plot a histogram of word frequency.
    :param word_count_dict: Dictionary containing word-frequency pairs.
    :param hist_size: Number of words to plot in the histogram.
    """
    words = list(word_count_dict.keys())[:hist_size]
    frequencies = list(word_count_dict.values())[:hist_size]
    plt.figure(figsize=(12, 8))
    plt.bar(words, frequencies, color='skyblue')
    plt.title('Word Frequency', fontsize=16)
    plt.xlabel('Words', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    path_out = '../outputs/word_frequency.jpg'
    plt.savefig(path_out)
    plt.show()
    print(f'Saved word frequency chart: {path_out}')

def save_training_results(loss_history, batch_loss_history, hyperparameters, output_dir):
    # Plot and save epoch loss history
    plt.figure()
    plt.plot(range(len(loss_history)), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History (per Epoch)')
    plt.savefig(f'{output_dir}/loss_history.png')
    plt.close()

    # Plot and save batch loss history
    plt.figure()
    plt.plot(range(len(batch_loss_history)), batch_loss_history, marker='o', markersize=2)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss History (per Batch)')
    plt.savefig(f'{output_dir}/batch_loss_history.png')
    plt.close()

    # Save hyperparameters and training time
    with open(f'{output_dir}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)
