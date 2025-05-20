# Sequence-to-Sequence Transliteration Model

This project implements a sequence-to-sequence (Seq2Seq) model with attention for transliteration tasks, specifically demonstrated on the Dakshina dataset (English to Indic languages). It uses PyTorch Lightning Fabric for efficient distributed training and Weights & Biases (W&B) for experiment tracking, logging, and visualization.

## Features

- **Seq2Seq Architecture:** Implements an Encoder-Decoder model using RNNs (GRU, LSTM, or RNN cells).
- **Attention Mechanism:** Supports Bahdanau attention for improved context handling in long sequences.
- **Distributed Training:** Leverages PyTorch Lightning Fabric for multi-GPU and mixed-precision training.
- **Weights & Biases Integration:**
  * Logs training/validation metrics (loss, accuracy, F1-score).
  * Tracks hyperparameters and system metrics.
  * Logs example attention heatmaps during validation.
  * Saves all test set predictions to a CSV file.
  * Uploads a sample of test predictions as a W&B Table.
  * Generates and logs a character-level confusion matrix for the test set.
  * Logs a "Focused Attention Visualization" for interpretability on test samples.
- **Hyperparameter Sweeps:** Integrated with W&B Sweeps for automated hyperparameter optimization.
- **Checkpointing:** Saves the best model based on validation loss, top-K checkpoints, and the last epoch's model.
- **Customizable:** Model architecture (embedding dimensions, hidden dimensions, number of layers, RNN cell type, dropout) and training parameters (batch size, learning rate, epochs, teacher forcing) are configurable via command-line arguments.
- **Data Handling:**
  * Uses a character-level tokenizer.
  * Includes robust data loading for Dakshina dataset format.
  * Automatically splits a validation set from training data if no separate validation set is provided.
- **Reproducibility:** Supports setting a random seed.

## Project Structure (Assumed)

```
.
├── train.py                         # Main training script
├── tokenizer.py                     # Character tokenizer implementation
├── dataset.py                       # PyTorch Dataset and Dataloader utilities
├── model.py                         # Seq2Seq model (Encoder, Decoder, Attention) definitions
├── utils.py                         # Utility functions (e.g., plotting, beam search)
├── TiroDevanagariHindi-Regular.ttf  # Font file for visualizations
├── test_predictions.csv             # Output: All predictions on the test set
├── confusion_matrix.jpg             # Output: Saved confusion matrix image
├── checkpoints/                     # Output: Saved model checkpoints (if enabled)
├── sweep_config.yaml                # Example: W&B sweep configuration file (user-created)
└── README.md                        # This file
```

## Setup

1. **Clone the repository (if applicable).**
2. **Create a Conda/Virtual Environment (Recommended):**
   ```bash
   conda create -n translit python=3.9
   conda activate translit
   ```
3. **Install Dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust CUDA version if needed
   pip install lightning wandb pandas scikit-learn matplotlib seaborn tqdm pyyaml
   ```
4. **Dataset:**
   * Download the Dakshina dataset (or your target transliteration dataset).
   * Ensure it's in a directory with `train.tsv`, `dev.tsv` (optional), and `test.tsv` files. The script attempts to find files like `{lang_code}_train.tsv` or `train.tsv`.
5. **Font for Visualization:**
   * Place the `TiroDevanagariHindi-Regular.ttf` font file (or another Devanagari-supporting .ttf font) in the root directory of the project, or ensure it's installed system-wide and Matplotlib can find it by name. The script is currently configured to look for this specific file in the current directory for some plots.
6. **Weights & Biases Account:**
   * Sign up for a free W&B account at [wandb.ai](https://wandb.ai).
   * Log in from your terminal: `wandb login`

## Usage

### Training a Single Model

Execute the `train.py` script with appropriate command-line arguments.

**Example:**

```bash
python train.py \
    --dataset_path /path/to/your/dakshina_dataset_hi_v1.0 \
    --project_name "my_translit_project" \
    --run_name "hi_gru_attention_run1" \
    --rnn_cell_type GRU \
    --attention_type bahdanau \
    --hidden_dim 256 \
    --num_encoder_layers 2 \
    --num_decoder_layers 1 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --num_epochs 20 \
    --dropout 0.2 \
    --checkpoint_dir ./my_checkpoints \
    --mixed_precision bf16-mixed
```

**Key Command-Line Arguments:**

- `--dataset_path`: Path to the dataset directory.
- `--project_name`: W&B project name.
- `--run_name`: (Optional) Custom name for the W&B run.
- `--disable_wandb`: To run without W&B logging.
- Model architecture args: `--input_embed_dim`, `--hidden_dim`, `--num_encoder_layers`, `--num_decoder_layers`, `--rnn_cell_type`, `--attention_type`, etc.
- Training args: `--num_epochs`, `--batch_size`, `--learning_rate`, `--teacher_forcing_ratio`, etc.
- `--checkpoint_dir`: Directory to save model checkpoints.
- `--mixed_precision`: Fabric mixed precision setting (32, 16-mixed, bf16-mixed).
- `--seed`: Random seed for reproducibility.

Run `python train.py --help` for a full list of arguments.

### Running a W&B Sweep (Hyperparameter Optimization)

1. **Create a Sweep Configuration File** (e.g., `sweep_config.yaml`):
   Define the search strategy, parameters, and their ranges/values. Example:

```yaml
method: bayes # or random, grid
metric:
  name: val/acc # Or val/loss, val/f1_macro
  goal: maximize # Or minimize for loss
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  hidden_dim:
    values: [128, 256, 512]
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
  num_encoder_layers:
    values: [1, 2]
  num_decoder_layers:
    values: [1, 2]
  label_smoothing:
    distribution: uniform
    min: 0.0
    max: 0.2
# You can add other fixed parameters from train.py here if needed
# command:
#  - ${env}
#  - python
#  - ${program}
#  - ${args}
#  - --dataset_path=/path/to/your/dataset # Fixed path
```

2. **Initialize the Sweep with W&B:**

```bash
wandb sweep sweep_config.yaml -p your_wandb_project_name
```

This will output a sweep ID.

3. **Run W&B Agents:**
   Use the sweep ID from the previous step. The agent will pick hyperparameters from the sweep configuration and run train.py.

```bash
# Replace <YOUR_ENTITY/YOUR_PROJECT/SWEEP_ID> with the actual sweep ID
# Example: wandb agent jdoe/my_translit_project/abcdef12
wandb agent <YOUR_ENTITY/YOUR_PROJECT/SWEEP_ID>
```

Alternatively, you can pass the sweep ID directly to the script:

```bash
python train.py \
    --dataset_path /path/to/your/dakshina_dataset_hi_v1.0 \
    --project_name "my_translit_project" \
    --sweep_id <SWEEP_ID_FROM_WANDB_UI_OR_CLI>
```

Or pass the config path to let the script initialize the sweep:

```bash
python train.py \
    --dataset_path /path/to/your/dakshina_dataset_hi_v1.0 \
    --project_name "my_translit_project" \
    --sweep_config_path ./sweep_config.yaml
```

## Output Visualizations on W&B

During and after training, the following visualizations will be available in your W&B dashboard:

- **Metrics**: Training loss, validation loss, token accuracy, exact match accuracy, F1 macro score over epochs.
- **Attention Heatmap (Standard)**: For a sample from the validation set, logged periodically. Shows alignment between source and predicted tokens.
- **Focused Attention Visualization (Test Set)**: For a sample from the test set, this plot shows which input characters the model paid most attention to when generating each output character. Brighter colors mean higher attention.
- **Test Predictions Sample Table**: A table with 20 random samples from the test set, showing source, true target, and model prediction.
- **Test Set Confusion Matrix**: A character-level confusion matrix (normalized percentages) showing common misclassifications on the test set.

## Code Modules

- **tokenizer.py**: Handles character-level tokenization, mapping characters to indices and vice-versa.
- **dataset.py**: Defines TransliterationDataset for loading data and collate_fn for batching sequences with padding.
- **model.py**: Contains the PyTorch nn.Module definitions for:
  * **EncoderRNN**: Encodes the source sequence.
  * **DecoderRNN**: Generates the target sequence, potentially using attention.
  * **BahdanauAttention**: Implements the Bahdanau attention mechanism.
  * **Seq2Seq**: The main model orchestrating the encoder and decoder.
- **utils.py**:
  * **plot_attention_heatmap**: Generates the standard attention heatmap visualization.
  * **log_focused_attention_to_wandb**: Generates and logs the focused attention visualization.
  * **beam_search_decode**: (If used) Implements beam search decoding for inference.