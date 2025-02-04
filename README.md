# nlp_assignment3

Sure! Here's a `README.md` file for your GitHub repository.

```markdown
# English to French Translator with Attention Variants

This project is a comprehensive implementation of an English to French translation model using three different attention mechanisms: Multiplicative, General, and Additive. The project uses PyTorch and the `datasets` library to load and preprocess the data, and includes model training, evaluation, and visualization of attention weights.

## Features

- Train and evaluate translation models with Multiplicative, General, and Additive attention mechanisms.
- Visualize attention weights for translations.
- Calculate BLEU scores for model evaluation.
- Support for mixed precision training using GradScaler.
- Tokenization and vocabulary building for English and French texts.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/english-french-translator.git
    cd english-french-translator
    ```

2. **Create a virtual environment and activate it**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pre-trained models and place them in the project directory**:

    - `en-fr-transformer-multiplicative.pt`
    - `en-fr-transformer-general.pt`
    - `en-fr-transformer-additive.pt`

## Usage

To start the training and evaluation process, run the following command:

```bash
python app.py
```

This script will load the dataset, create tokenizers, initialize the models, and start the training and evaluation process. The results, including loss plots and training metrics, will be saved in the project directory.

## How to Use

1. **Tokenization**:
    - The `CustomTokenizer` class is used to tokenize and build vocabularies for English and French texts.
    - The tokenizers are initialized with the training data from the dataset.

2. **Training**:
    - The `train` function performs training with gradient accumulation and mixed precision.
    - The `evaluate` function evaluates the model on the validation set.

3. **Visualization**:
    - The `visualize_attention` function creates heatmaps of attention weights for given source and target text pairs.

4. **Evaluation**:
    - The `calculate_bleu` function calculates the BLEU score for the model's predictions.
    - The `evaluate_translations` function evaluates model translations on test set examples.

## File Structure

- `app.py`: The main script for training, evaluating, and visualizing the models.
- `requirements.txt`: The list of required dependencies.
- `en-fr-transformer-multiplicative.pt`: Pre-trained Multiplicative Attention model file.
- `en-fr-transformer-general.pt`: Pre-trained General Attention model file.
- `en-fr-transformer-additive.pt`: Pre-trained Additive Attention model file.
- `attention_results.csv`: CSV file containing the training results for each attention variant.
- `loss_plot_multiplicative.png`: Loss plot for Multiplicative Attention.
- `loss_plot_general.png`: Loss plot for General Attention.
- `loss_plot_additive.png`: Loss plot for Additive Attention.

## Dependencies

- `torch`
- `datasets`
- `pandas`
- `matplotlib`
- `seaborn`
- `sacrebleu`
- `tqdm`

## License

This project is licensed under the MIT License.

## Acknowledgements

- Hugging Face for providing the `datasets` library.
- PyTorch for the deep learning framework.
- Streamlit for the web application framework.

If you have any questions or need further assistance, feel free to ask!

```
Certainly! Here's the continuation of your `README.md` file to include details about the attention maps evaluation and translation testing:

```markdown
## Evaluation and Visualization

The script includes functionality to evaluate attention maps and test model translations. The attention maps provide insights into how the models focus on different parts of the input sentence during translation.

### Attention Maps Evaluation

The `evaluate_attention_maps` function evaluates and visualizes attention maps for trained models with detailed debugging information. It processes a set of test pairs and generates attention visualizations for each pair.

### Translation Testing

The script also includes functionality to test model translations on custom English sentences. This helps verify the quality and accuracy of translations produced by the models.

## Running the Evaluation

To run the evaluation and generate attention maps, simply execute the script:

```bash
python app.py
```

### Example Output

The script will evaluate translations and generate attention maps for the specified models. Example output includes:

- Attention maps saved as PNG files.
- Translation results printed to the console.

### Attention Maps

For each attention variant (Multiplicative and General), the script generates attention maps for predefined test pairs. The attention maps are saved as PNG files with filenames indicating the attention variant and the first few words of the source sentence.

### Translation Results

The script tests translations on a set of custom English sentences and prints the translation results to the console.

### Sample Output

```
=== Attention Visualization ===

Evaluating multiplicative attention model:

Generating visualizations...

Processing pair:
English: The time now is 05:08 .
French: The time now is 05:05 .

Raw tokens:
Source tokens: [2, 4, 96, 149, 11, 0, 172, 3]
Target tokens: [2, 1035, 6327, 6869, 2256, 0, 167, 3]

Decoded tokens before filtering:
Source tokens: ['<sos>', 'the', 'time', 'now', 'is', '<unk>', '05:08', '<eos>']
Target tokens: ['<sos>', 'le', 'temps', 'actuel', 'est', '<unk>', '05:05', '<eos>']

Tokens after filtering:
Source tokens: ['the', 'time', 'now', 'is', '05:08']
Target tokens: ['le', 'temps', 'actuel', 'est', '05:05']

Attention shape: (5, 5)

Saved attention map to: attention_map_multiplicative_The_time_now_i.png

...

=== Testing translations ===

English: Hello, how are you doing today?
French: bonjour, comment ça va aujourd'hui ?

English: I love learning new languages.
French: J'adore apprendre de nouvelles langues.

...

Evaluation complete! Check the generated visualizations and translation results.
```

## Dependencies

- `torch`
- `datasets`
- `pandas`
- `matplotlib`
- `seaborn`
- `sacrebleu`
- `tqdm`

## License

This project is licensed under the MIT License.

## Acknowledgements

- Hugging Face for providing the `datasets` library.
- PyTorch for the deep learning framework.
- Matplotlib and Seaborn for visualization tools.

If you have any questions or need further assistance, feel free to ask!
```

This continuation includes detailed information about the evaluation and visualization functionalities, sample output, and how to run the script to generate attention maps and test translations.
Feel free to customize the repository URL, license, and any other sections as needed. If you have any questions or need further assistance, feel free to ask!
