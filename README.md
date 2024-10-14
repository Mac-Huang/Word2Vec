# Skip-Gram Model Training Project

This project provides an implementation of a Skip-Gram model for learning word embeddings from text data. The Skip-Gram model is trained using a dataset to predict context words given a target word, helping capture word similarities and semantic relationships. This project includes all necessary scripts and a tutorial Jupyter Notebook to guide you through the training process step-by-step.

## Project Structure

```
skip_gram_project/
|-- data/
|   |-- raw_data.txt               # Raw text data used for training
|   |-- processed_data.pkl         # Processed version of the raw text data
|
|-- src/
|   |-- __init__.py                # Initialization file
|   |-- build_vocab.py             # Data preprocessing module
|   |-- skip_gram.py               # Skip-Gram model implementation
|   |-- train.py                   # Training logic for the Skip-Gram model
|   |-- utils.py                   # Utility functions for saving results and handling data
|   |-- evaluate.py                # Evaluation logic for the model
|
|-- outputs/
|   |-- model.pth                 # Trained model parameters
|   |-- loss_history.png          # Visualization of the training loss per epoch
|   |-- batch_loss_history.png    # Visualization of the training loss per batch
|   |-- word2id_dict.npy          # Mapping of words to indices
|   |-- id2word_dict.npy          # Mapping of indices to words
|   |-- word_count.pkl            # Saved word frequency data
|   |-- hyperparameters.json      # Training hyperparameters and duration
|
|-- tests/
|   |-- test_data_preprocessing.py  # Test code for data preprocessing
|   |-- test_model.py              # Test code for model functions
|
|-- requirements.txt              # Project dependencies
|-- README.md                     # Project documentation (this file)
```

## Prerequisites

To run this project, you will need:
- Python 3.7 or higher
- PyTorch
- NumPy
- Matplotlib
- tqdm

All dependencies are listed in `requirements.txt`. You can install them by running:

```sh
pip install -r requirements.txt
```

## Training the Skip-Gram Model

### Step 1: Data Preparation
Place your raw text data into the `data/` folder (e.g., `raw_data.txt`). Run `build_vocab.py` to preprocess the data, create word dictionaries, and generate the frequency data required for training.

### Step 2: Running the Training
To train the Skip-Gram model, you can either run the script `train.py` directly from the command line:

```sh
python src/train.py
```

Alternatively, you can follow the step-by-step training process provided in the Jupyter Notebook `notebooks/skip_gram_tutorial_notebook.ipynb`. This notebook provides a detailed walkthrough for setting up the model, training it, and analyzing the results.

### Step 3: Saving and Analyzing Results
During training, the model parameters are saved in the `outputs/` folder as `model.pth`. Additionally, the loss history during training is saved as `loss_history.png` (per epoch) and `batch_loss_history.png` (per batch).

The `utils.py` script includes helper functions that save the training results, hyperparameters, and visualizations.

## Evaluation
You can evaluate the trained embeddings using `evaluate.py`. The evaluation script tests the quality of the word embeddings by measuring similarity on predefined word pairs or by visualizing using dimensionality reduction techniques like PCA or t-SNE.

## Visualizations
Loss curves are saved as images in the `outputs/` folder:
- `loss_history.png`: Shows the average loss per epoch.
- `batch_loss_history.png`: Shows the loss per batch during training.

These visualizations help you understand the model's convergence behavior and training dynamics.

## How to Use the Learned Embeddings
Once trained, the word embeddings can be loaded and used in downstream tasks like:
- Document classification
- Sentiment analysis
- Information retrieval

The embedding weights can be extracted using:

```python
embedding_weights = torch.load('../outputs/model.pth')
```

You can also use the `get_input_embedding()` function from the `Word2Vec` model to retrieve the learned embeddings for further analysis or downstream use.

## Contribution
Feel free to fork this repository and improve upon it. Contributions are welcome for expanding the training options, implementing additional evaluation methods, or optimizing training performance.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
If you have questions or feedback, please open an issue in the GitHub repository, or contact the project maintainers directly.

Happy embedding!