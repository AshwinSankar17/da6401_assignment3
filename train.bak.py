import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from functools import partial
import os
import time
import yaml
import wandb
import numpy as np
import argparse
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm

from wandb.integration.lightning.fabric import WandbLogger
from sklearn.metrics import f1_score

# Assuming these modules are in the same directory or accessible via PYTHONPATH
from tokenizer import CharTokenizer
from dataset import TransliterationDataset, collate_fn
from model import EncoderRNN, DecoderRNN, Seq2Seq, BahdanauAttention
from utils import plot_attention_heatmap, beam_search_decode


def load_dakshina_split(file_path: Optional[str]) -> Tuple[List[str], List[str]]:
    """
    Loads a Dakshina dataset split from a TSV file.

    Assumes 3 columns in the TSV: source_word, target_word, attestations.
    Only the first two columns (source and target words) are used.
    Performs preprocessing:
        - Drops rows if source or target is NaN or not a string.
        - Drops rows if source or target is exactly "</s>" or empty after stripping whitespace.

    Args:
        file_path (str, optional): Path to the .tsv file. If None, returns empty lists.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing (source_sentences, target_sentences).
    """
    if file_path is None:
        return [], []

    source_sents: List[str] = []
    target_sents: List[str] = []
    try:
        df = pd.read_csv(
            file_path, sep="\t", header=None, on_bad_lines="skip", quoting=3, dtype=str
        )

        if df.shape[1] < 2:
            # fabric.print might not be available here, use standard print for this util
            print(
                f"Warning ({os.path.basename(file_path if file_path else 'None')}): Less than 2 columns found. Returning empty lists."
            )
            return [], []

        df = df.iloc[:, [0, 1]].copy()
        # Assuming the Dakshina format has native words (target) in the first column
        # and English/source words in the second, as per typical transliteration tasks.
        # If it's the other way, this should be df.columns = ["source", "target"]
        df.columns = [
            "target",
            "source",
        ]  # Original: ["target", "source"] - check if this is correct for Dakshina format

        initial_rows = len(df)
        df.dropna(subset=["source", "target"], inplace=True)
        df = df[
            df["source"].apply(isinstance, args=(str,))
            & df["target"].apply(isinstance, args=(str,))
        ]

        df["source_stripped"] = df["source"].str.strip()
        df["target_stripped"] = df["target"].str.strip()

        df = df[~((df["source_stripped"] == "</s>") | (df["source_stripped"] == ""))]
        df = df[~((df["target_stripped"] == "</s>") | (df["target_stripped"] == ""))]

        if (
            initial_rows > 0
            and len(df) < initial_rows
            and (len(df) * 1.0 / initial_rows < 0.5)
        ):
            print(
                f"Warning ({os.path.basename(file_path if file_path else 'None')}): Significant rows dropped during preprocessing. Initial: {initial_rows}, Final: {len(df)}"
            )

        source_sents = df["source"].tolist()
        target_sents = df["target"].tolist()

    except FileNotFoundError:
        print(f"Warning: Data file not found {file_path}. Returning empty lists.")
    except Exception as e:
        print(
            f"Error reading or processing data file {file_path}: {e}. Returning empty lists."
        )

    if not source_sents or not target_sents:
        # fabric.print might not be available here
        # print(f"Warning ({os.path.basename(file_path if file_path else 'None')}): No data loaded after processing. Returning empty lists.")
        return [], []

    return source_sents, target_sents


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    num_classes: int,
    set_name: str = "Validation",  # set_name is kept for potential future use, but F1 is now always calculated
) -> Tuple[float, float, float]:
    """
    Calculates token accuracy, exact match accuracy, and macro F1 score.

    Args:
        predictions (torch.Tensor): Predicted token IDs (Batch, SeqLen).
        targets (torch.Tensor): Ground truth token IDs (Batch, SeqLen).
        pad_idx (int): Index of the PAD token.
        num_classes (int): Total number of classes (vocabulary size) for F1 calculation.
        set_name (str): Name of the dataset split (e.g., "Validation", "Test").

    Returns:
        Tuple[float, float, float]: token_accuracy, exact_match_accuracy, f1_macro_score
    """
    max_len = min(predictions.size(1), targets.size(1))
    predictions_sliced = predictions[:, :max_len]
    targets_sliced = targets[:, :max_len]

    mask = targets_sliced != pad_idx
    correct_tokens = ((predictions_sliced == targets_sliced) & mask).sum().item()
    total_relevant_tokens = mask.sum().item()
    token_acc = (
        correct_tokens / total_relevant_tokens if total_relevant_tokens > 0 else 0.0
    )

    exact_matches = 0
    for i in range(targets_sliced.size(0)):
        if mask[i].any():  # If target has non-PAD tokens
            pred_seq = predictions_sliced[i][mask[i]]
            tgt_seq = targets_sliced[i][mask[i]]
            if pred_seq.tolist() == tgt_seq.tolist():
                exact_matches += 1
        elif not predictions_sliced[i][predictions_sliced[i] != pad_idx].any():
            # Target is all PADs, and prediction is also all PADs (or empty)
            exact_matches += 1

    exact_match_acc = (
        exact_matches / targets_sliced.size(0) if targets_sliced.size(0) > 0 else 0.0
    )

    f1_macro = 0.0
    # Calculate F1 score for all sets now, not just Test
    y_true_flat = targets_sliced[mask].cpu().numpy()
    y_pred_flat = predictions_sliced[mask].cpu().numpy()

    if len(y_true_flat) > 0 and len(y_pred_flat) > 0:
        f1_macro = f1_score(
            y_true_flat,
            y_pred_flat,
            average="macro",
            labels=np.arange(num_classes),  # Ensure all possible labels are considered
            zero_division=0,
        )

    return token_acc, exact_match_acc, f1_macro


@torch.no_grad()
def evaluate_on_dataloader(
    fabric: L.Fabric,
    model: Seq2Seq,
    dataloader: DataLoader,
    criterion: nn.Module,
    target_tokenizer: CharTokenizer,
    set_name: str = "Validation",
) -> Tuple[float, float, float, float]:
    """
    Evaluates the model on a given dataloader (e.g., validation or test).
    Calculates loss, token accuracy, exact match accuracy, and F1 macro score.
    Does not perform backpropagation.
    """
    model.eval()
    (
        total_loss_sum,
        total_token_acc_sum,
        total_exact_match_acc_sum,
        total_f1_macro_sum,
        num_batches,
    ) = (0.0, 0.0, 0.0, 0.0, 0)

    for src_seqs, src_lengths, trg_seqs, trg_lengths in dataloader:
        # decoder_logits: (output_seq_len, batch_size, vocab_size)
        decoder_logits, _ = model(
            src_seqs,
            src_lengths,
            trg_seqs,
            teacher_forcing_ratio=0.0,  # teacher_forcing_ratio=0.0 for eval
        )

        # For loss:
        # preds_for_loss: (output_seq_len * batch_size, vocab_size)
        # targets_for_loss: (output_seq_len * batch_size)
        # trg_seqs is (trg_len, batch_size). Target for loss should exclude SOS.
        # Max length for comparison is decoder_logits.shape[0]
        output_len = decoder_logits.shape[0]
        preds_for_loss = decoder_logits.reshape(-1, decoder_logits.shape[-1])
        targets_for_loss = trg_seqs[1 : output_len + 1].reshape(-1)

        current_batch_loss = float("nan")

        if preds_for_loss.numel() > 0 and targets_for_loss.numel() > 0:
            loss = criterion(preds_for_loss, targets_for_loss)
            current_batch_loss = loss.item()
            if not np.isnan(current_batch_loss):  # Ensure we only add valid numbers
                total_loss_sum += current_batch_loss

            # For metrics:
            # predicted_tokens_batch: (batch_size, output_seq_len)
            # targets_for_metrics: (batch_size, output_seq_len)
            predicted_tokens_batch = decoder_logits.argmax(dim=2).permute(1, 0)
            targets_for_metrics = trg_seqs[1 : output_len + 1].permute(1, 0)

            token_acc, exact_match_acc, f1_macro = calculate_metrics(
                predicted_tokens_batch,
                targets_for_metrics,
                target_tokenizer.PAD_IDX,
                target_tokenizer.vocab_size,
                set_name=set_name,
            )

            total_token_acc_sum += token_acc
            total_exact_match_acc_sum += exact_match_acc
            total_f1_macro_sum += f1_macro

        num_batches += 1

    avg_loss = total_loss_sum / num_batches if num_batches > 0 else float("nan")
    avg_token_acc = (
        total_token_acc_sum / num_batches if num_batches > 0 else float("nan")
    )
    avg_exact_match_acc = (
        total_exact_match_acc_sum / num_batches if num_batches > 0 else float("nan")
    )
    avg_f1_macro = total_f1_macro_sum / num_batches if num_batches > 0 else float("nan")

    return avg_loss, avg_token_acc, avg_exact_match_acc, avg_f1_macro


def train_and_evaluate(fabric: L.Fabric, args: argparse.Namespace):
    """
    Main training and evaluation process.
    Orchestrates data loading, model setup, training loop, validation, checkpointing,
    and final test set evaluation.
    """
    fabric.seed_everything(args.seed)

    # --- 1. Load Data ---
    if not args.dataset_path or not os.path.isdir(args.dataset_path):
        message = f"Error: Dataset path '{args.dataset_path}' is not a valid directory."
        fabric.print(message)
        raise FileNotFoundError(message)

    all_files_in_path = os.listdir(args.dataset_path)
    # Prioritize files with language code from dataset_path if possible, then generic
    # Example: if dataset_path is /path/to/dakshina_dataset_mr_v1.0 (for Marathi)
    # Look for mr_train.tsv, then train.tsv
    lang_code_from_path = ""
    try:
        # Extract language code like 'hi' from 'dakshina_dataset_hi_v1.0'
        parts = os.path.basename(args.dataset_path.strip(os.sep)).split("_")
        if len(parts) > 2 and len(parts[2]) == 2:  # Simple check for lang code
            lang_code_from_path = parts[2]
    except Exception:
        pass  # lang_code_from_path remains ""

    def find_file(suffix):
        if lang_code_from_path:
            specific_name = f"{lang_code_from_path}_{suffix}.tsv"
            if specific_name in all_files_in_path:
                return specific_name
        generic_name = f"{suffix}.tsv"  # Original file name had {lang}.{split}.tsv
        # The prompt file names are like {lang}_{split}.tsv or just {split}.tsv
        # The code expects {split}.tsv e.g., train.tsv
        # Let's stick to current logic: look for file ending with {split}.tsv
        found_file = next(
            (f for f in all_files_in_path if f.endswith(f"{suffix}.tsv")), None
        )
        if found_file:
            return found_file
        # Fallback for common pattern like {lang}.{split}.tsv (e.g. en.train.tsv)
        if lang_code_from_path:  # Try to use lang code if available
            found_file = next(
                (
                    f
                    for f in all_files_in_path
                    if f == f"{lang_code_from_path}.{suffix}.tsv"
                ),
                None,
            )
            if found_file:
                return found_file
        # If still not found, try any file matching *.suffix.tsv
        found_file = next(
            (f for f in all_files_in_path if f.endswith(f".{suffix}.tsv")), None
        )
        return found_file

    train_file_name = find_file("train")
    dev_file_name = find_file("dev")
    test_file_name = find_file("test")

    train_file = (
        os.path.join(args.dataset_path, train_file_name) if train_file_name else None
    )
    dev_file = os.path.join(args.dataset_path, dev_file_name) if dev_file_name else None
    test_file = (
        os.path.join(args.dataset_path, test_file_name) if test_file_name else None
    )

    fabric.print(
        f"Attempting to load training data from: {train_file or 'Not found/specified'}"
    )
    train_source_sents, train_target_sents = load_dakshina_split(train_file)
    fabric.print(
        f"Attempting to load validation data from: {dev_file or 'Not found/specified'}"
    )
    dev_source_sents, dev_target_sents = load_dakshina_split(dev_file)
    fabric.print(
        f"Attempting to load test data from: {test_file or 'Not found/specified'}"
    )
    test_source_sents, test_target_sents = load_dakshina_split(test_file)

    if not train_source_sents:
        message = f"Fatal Error: No training data loaded from '{train_file}'. Exiting."
        fabric.print(message)
        raise ValueError(message)

    # --- 2. Tokenizers ---
    source_tokenizer = CharTokenizer()
    source_tokenizer.fit(train_source_sents)
    target_tokenizer = CharTokenizer()
    target_tokenizer.fit(train_target_sents)

    if fabric.global_rank == 0 and args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        source_tokenizer.save_tokenizer(
            os.path.join(args.checkpoint_dir, "source_tokenizer.json")
        )
        target_tokenizer.save_tokenizer(
            os.path.join(args.checkpoint_dir, "target_tokenizer.json")
        )
    source_vocab_size, target_vocab_size = (
        source_tokenizer.vocab_size,
        target_tokenizer.vocab_size,
    )
    if fabric.global_rank == 0:
        fabric.print(f"Src vocab: {source_vocab_size}, Trg vocab: {target_vocab_size}")
        if wandb.run is not None:
            wandb.config.update(
                {
                    "source_vocab_size_runtime": source_vocab_size,
                    "target_vocab_size_runtime": target_vocab_size,
                },
                allow_val_change=True,
            )

    # --- 3. Datasets and DataLoaders ---
    train_dataset_obj = TransliterationDataset(
        train_source_sents,
        train_target_sents,
        source_tokenizer,
        target_tokenizer,
        verbose=(fabric.global_rank == 0),
    )
    val_dataset_obj = None
    if dev_source_sents:
        val_dataset_obj = TransliterationDataset(
            dev_source_sents,
            dev_target_sents,
            source_tokenizer,
            target_tokenizer,
            verbose=(fabric.global_rank == 0),
        )
    elif dev_file and not dev_source_sents:  # dev_file was specified but loaded no data
        fabric.print(f"Warning: Dev file {dev_file} was found but loaded no data.")

    if (
        val_dataset_obj is None and len(train_dataset_obj) >= 10
    ):  # only split if train is reasonably sized
        fabric.print(
            "No validation data loaded or validation data is empty, attempting to split from training set."
        )
        original_train_len = len(train_dataset_obj)
        # Ensure val split is at least 1, or a certain percentage, e.g., 10%
        val_len_from_train = max(1, int(0.1 * original_train_len))
        train_len_for_split = original_train_len - val_len_from_train

        if val_len_from_train > 0 and train_len_for_split > 0:
            train_dataset_obj_split, val_dataset_obj_split = (
                torch.utils.data.random_split(
                    train_dataset_obj,
                    [train_len_for_split, val_len_from_train],
                    generator=torch.Generator().manual_seed(args.seed),
                )
            )
            train_dataset_obj = train_dataset_obj_split
            val_dataset_obj = val_dataset_obj_split  # This is now a Subset object
            fabric.print(
                f"Created validation split: {len(train_dataset_obj)} train, {len(val_dataset_obj)} val samples."
            )
        else:
            fabric.print(
                "Could not create validation split from train (train set too small or split resulted in zero validation samples). Validation will be skipped."
            )
            val_dataset_obj = None  # Ensure it's None
    elif val_dataset_obj is None:  # No dev data and train set too small for split
        fabric.print(
            "No validation data provided and training set too small for split. Validation will be skipped."
        )

    custom_collate_fn = partial(
        collate_fn,
        source_pad_idx=source_tokenizer.PAD_IDX,
        target_pad_idx=target_tokenizer.PAD_IDX,
    )
    num_cpus = os.cpu_count() or 1
    # fabric.world_size is at least 1
    dataloader_num_workers = min(
        max(0, args.num_workers), num_cpus // fabric.world_size
    )
    if fabric.global_rank == 0:
        fabric.print(f"Using {dataloader_num_workers} dataloader workers per process.")

    train_dataloader = DataLoader(
        train_dataset_obj,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle is handled by DistributedSampler if used by Fabric
        collate_fn=custom_collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),  # pin_memory if CUDA is available
        drop_last=fabric.world_size > 1,  # drop_last if distributed
    )
    val_dataloader = None
    if val_dataset_obj and len(val_dataset_obj) > 0:
        val_dataloader = DataLoader(
            val_dataset_obj,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    test_dataloader = None
    if test_source_sents and test_target_sents:
        test_dataset_obj = TransliterationDataset(
            test_source_sents,
            test_target_sents,
            source_tokenizer,
            target_tokenizer,
            verbose=False,  # Less verbose for test set
        )
        if len(test_dataset_obj) > 0:
            test_dataloader = DataLoader(
                test_dataset_obj,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=dataloader_num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            fabric.print("Test data was found but resulted in an empty dataset object.")

    train_dataloader = fabric.setup_dataloaders(
        train_dataloader, use_distributed_sampler=True
    )  # Default is True
    if val_dataloader:
        val_dataloader = fabric.setup_dataloaders(
            val_dataloader, use_distributed_sampler=False
        )
    if test_dataloader:
        test_dataloader = fabric.setup_dataloaders(
            test_dataloader, use_distributed_sampler=False
        )

    # --- 4. Model ---
    num_enc_layers = (
        args.num_encoder_layers
        if args.num_encoder_layers is not None
        else args.num_layers
    )
    num_dec_layers = (
        args.num_decoder_layers
        if args.num_decoder_layers is not None
        else args.num_layers
    )

    encoder = EncoderRNN(
        source_vocab_size,
        args.input_embed_dim,
        args.hidden_dim,
        num_enc_layers,
        args.rnn_cell_type,
        args.dropout,
        args.encoder_bidirectional,
    )
    enc_out_dim_attn = args.hidden_dim * (2 if args.encoder_bidirectional else 1)
    attn_module = (
        BahdanauAttention(enc_out_dim_attn, args.hidden_dim, args.attention_dim)
        if args.attention_type.lower() == "bahdanau"
        else None
    )
    decoder = DecoderRNN(
        target_vocab_size,
        args.target_embed_dim,
        args.hidden_dim,
        num_dec_layers,
        args.rnn_cell_type,
        args.dropout,
        attn_module,
        enc_out_dim_attn if attn_module else None,
    )
    model = Seq2Seq(
        encoder,
        decoder,
        target_tokenizer.SOS_IDX,
        target_tokenizer.EOS_IDX,
        fabric.device,  # Pass fabric.device to model for internal tensor placement if needed
    )
    model = fabric.setup_module(model)

    if fabric.global_rank == 0 and wandb.run is not None:
        log_freq = 100
        if (
            hasattr(train_dataloader, "__len__") and len(train_dataloader) > 1
        ):  # Check if dataloader has length
            log_freq = min(
                100, len(train_dataloader) // 2 if len(train_dataloader) > 1 else 1
            )
        elif not hasattr(train_dataloader, "__len__"):  # Iterable dataset
            fabric.print(
                "Warning: Train dataloader has no __len__, using default wandb.watch log_freq=100."
            )

        wandb.watch(
            model,
            log="all",
            log_freq=log_freq,
        )

    # --- 5. Optimizer and Loss ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, fused=True)
    optimizer = fabric.setup_optimizers(optimizer)
    criterion = nn.CrossEntropyLoss(
        ignore_index=target_tokenizer.PAD_IDX, label_smoothing=args.label_smoothing
    )

    # --- 6. Training Loop ---
    if fabric.global_rank == 0:
        fabric.print("Starting training...")
    best_val_loss = float("inf")
    top_k_checkpoints: List[Tuple[float, int, str]] = []  # (val_loss, epoch, path)
    top_k = 3

    epoch_pbar = tqdm(
        range(args.num_epochs),
        desc="Epochs",
        unit="epoch",
        disable=(fabric.global_rank != 0),
        dynamic_ncols=True,
    )
    if fabric.global_rank == 0:
        initial_epoch_postfix = {"TrL": "N/A"}
        if val_dataloader:
            initial_epoch_postfix["VL"] = "N/A"
            initial_epoch_postfix["VAcc"] = "N/A"  # Exact Match Accuracy
            initial_epoch_postfix["VF1"] = "N/A"  # F1 Macro
        epoch_pbar.set_postfix(initial_epoch_postfix)

    for epoch in epoch_pbar:
        model.train()
        epoch_train_loss_sum = 0.0
        num_train_batches = 0
        start_time = time.time()
        avg_epoch_train_loss = float("nan")

        if not hasattr(train_dataloader, "__len__") or len(train_dataloader) == 0:
            if fabric.global_rank == 0:
                tqdm.write(
                    f"Epoch {epoch+1}: Training dataloader is empty or unsized. Skipping training for this epoch."
                )
        else:
            train_batch_pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1} Training",
                total=(
                    len(train_dataloader)
                    if hasattr(train_dataloader, "__len__")
                    else None
                ),
                disable=(fabric.global_rank != 0),
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )
            if fabric.global_rank == 0:
                train_batch_pbar.set_postfix(train_loss="N/A")

            for src_seqs, src_lengths, trg_seqs, trg_lengths in train_batch_pbar:
                optimizer.zero_grad()
                # trg_seqs for teacher forcing: (trg_len, batch_size)
                # decoder_logits: (trg_len - 1, batch_size, vocab_size)
                decoder_logits, _ = model(
                    src_seqs, src_lengths, trg_seqs, args.teacher_forcing_ratio
                )
                # For loss, preds: ( (trg_len-1)*batch, vocab_size ), targets: ( (trg_len-1)*batch )
                # Targets for loss should be trg_seqs[1:] aligned with decoder_logits
                output_len = decoder_logits.shape[0]  # Should be trg_seqs.shape[0] - 1
                preds_for_loss = decoder_logits.reshape(-1, decoder_logits.shape[-1])
                targets_for_loss = trg_seqs[1 : output_len + 1].reshape(-1)

                current_batch_loss = float("nan")
                if preds_for_loss.numel() == 0 or targets_for_loss.numel() == 0:
                    if fabric.global_rank == 0:
                        train_batch_pbar.set_postfix(train_loss="N/A (empty batch)")
                    continue

                loss = criterion(preds_for_loss, targets_for_loss)
                fabric.backward(loss)
                if args.grad_clip_val > 0:
                    fabric.clip_gradients(model, optimizer, clip_val=args.grad_clip_val)
                optimizer.step()

                current_batch_loss = loss.item()
                if not np.isnan(current_batch_loss):
                    epoch_train_loss_sum += current_batch_loss
                num_train_batches += 1

                fabric.log_dict({"train/batch_loss": loss})

                # if fabric.global_rank == 0 and wandb.run is not None:
                #     # Log step-wise train loss if not NaN
                #     if not np.isnan(current_batch_loss):
                #         wandb.log(
                #             {"train/batch_loss": current_batch_loss}
                #         )  # Renamed for clarity

                if fabric.global_rank == 0:
                    train_batch_pbar.set_postfix(
                        train_loss=(
                            f"{current_batch_loss:.3f}"
                            if not np.isnan(current_batch_loss)
                            else "N/A"
                        )
                    )

            avg_epoch_train_loss = (
                epoch_train_loss_sum / num_train_batches
                if num_train_batches > 0
                else float("nan")
            )
            if fabric.global_rank == 0 and hasattr(train_batch_pbar, "close"):
                train_batch_pbar.close()

        epoch_duration = time.time() - start_time

        (
            avg_epoch_val_loss,
            avg_val_token_acc,
            avg_val_exact_match_acc,
            avg_val_f1_macro,
        ) = (float("nan"), float("nan"), float("nan"), float("nan"))
        val_attention_maps_to_log = []

        if fabric.global_rank == 0:  # Validation only on rank 0
            if val_dataloader and (
                (hasattr(val_dataloader, "__len__") and len(val_dataloader) > 0)
                or not hasattr(val_dataloader, "__len__")
            ):
                (
                    avg_epoch_val_loss,
                    avg_val_token_acc,
                    avg_val_exact_match_acc,
                    avg_val_f1_macro,
                ) = evaluate_on_dataloader(
                    fabric,
                    model,
                    val_dataloader,
                    criterion,
                    target_tokenizer,
                    set_name="Validation",
                )

                if (
                    args.attention_type.lower() == "bahdanau"
                    and (epoch + 1) % args.log_attention_every_n_epochs == 0
                ):
                    # Use a single batch for attention visualization
                    try:
                        val_dl_iter = iter(val_dataloader)
                        (
                            src_seqs_val_attn,
                            src_lengths_val_attn,
                            trg_seqs_val_attn,
                            _,
                        ) = next(val_dl_iter)

                        # Ensure tensors are on the correct device for model input
                        src_seqs_val_attn = src_seqs_val_attn.to(fabric.device)
                        # src_lengths_val_attn should be on CPU for pack_padded_sequence
                        trg_seqs_val_attn = trg_seqs_val_attn.to(fabric.device)

                        # Get attention weights from model's inference mode
                        # Model's forward for inference: (src, src_len, None, teacher_forcing=0.0, max_len)
                        _, attention_weights_viz = model(
                            src_seqs_val_attn,
                            src_lengths_val_attn,  # Must be on CPU
                            None,  # No target sequence for inference visualization
                            0.0,
                            inference_max_len=trg_seqs_val_attn.size(0)
                            - 1,  # Max length based on example target
                        )

                        if (
                            attention_weights_viz is not None
                            and attention_weights_viz.numel() > 0
                        ):
                            # attention_weights_viz: (pred_len, batch_size, src_len)
                            # We need one example: (pred_len, src_len)
                            # Taking the first item in the batch
                            current_src_len_ex = src_lengths_val_attn[0].item()
                            single_attn_weights = (
                                attention_weights_viz[
                                    :, 0, :current_src_len_ex
                                ]  # (pred_len, current_src_len_ex)
                                .cpu()
                                .numpy()
                            )

                            # Source tokens
                            src_ids_list_ex = (
                                src_seqs_val_attn[:current_src_len_ex, 0].cpu().tolist()
                            )
                            src_tokens_list_ex = [
                                source_tokenizer.idx_to_char.get(
                                    i, "?"
                                )  # Use .get for safety
                                for i in src_ids_list_ex
                                if i < source_tokenizer.vocab_size
                                and i != source_tokenizer.PAD_IDX
                            ]

                            # Predicted tokens for this example using greedy approach from model
                            # Get actual model predictions for this batch
                            val_decoder_logits_viz, _ = model(
                                src_seqs_val_attn,
                                src_lengths_val_attn,  # CPU
                                None,  # No target sequence
                                0.0,  # Teacher forcing off
                                inference_max_len=trg_seqs_val_attn.size(0) - 1,
                            )

                            if val_decoder_logits_viz.numel() > 0:
                                pred_ids_for_heatmap_ex = (
                                    val_decoder_logits_viz.argmax(dim=2)[
                                        :, 0
                                    ]  # (pred_len)
                                    .cpu()
                                    .tolist()
                                )
                                pred_tokens_list_ex = []
                                for i in pred_ids_for_heatmap_ex:
                                    if i == target_tokenizer.EOS_IDX:
                                        break  # Stop at EOS
                                    if (
                                        i
                                        not in [
                                            target_tokenizer.PAD_IDX,
                                            target_tokenizer.SOS_IDX,
                                        ]
                                        and i < target_tokenizer.vocab_size
                                    ):
                                        pred_tokens_list_ex.append(
                                            target_tokenizer.idx_to_char.get(i, "?")
                                        )

                                # Ensure attention map matches length of predicted tokens if pred_tokens_list_ex is shorter
                                if (
                                    len(pred_tokens_list_ex)
                                    < single_attn_weights.shape[0]
                                ):
                                    single_attn_weights = single_attn_weights[
                                        : len(pred_tokens_list_ex), :
                                    ]

                                if (
                                    src_tokens_list_ex
                                    and pred_tokens_list_ex
                                    and single_attn_weights.size > 0
                                ):
                                    attn_map_img = plot_attention_heatmap(
                                        src_tokens_list_ex,
                                        pred_tokens_list_ex,
                                        single_attn_weights,
                                        f"Epoch {epoch+1} Val Attention Example",
                                    )
                                    if attn_map_img:
                                        val_attention_maps_to_log.append(attn_map_img)
                    except StopIteration:
                        fabric.print(
                            "Warning: Could not get batch from val_dataloader for attention viz (dataloader might be empty or too small)."
                        )
                    except Exception as e:
                        fabric.print(f"Error during attention visualization: {e}")

        
        log_dict = {
            "epoch": epoch + 1,
            "epoch_duration_secs": epoch_duration,
            "train/epoch_loss": avg_epoch_train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if val_dataloader:  # Only log val metrics if val_dataloader exists
            log_dict.update(
                {
                    "val/loss": avg_epoch_val_loss,
                    "val/token_accuracy": avg_val_token_acc,
                    "val/acc": avg_val_exact_match_acc,
                    "val/f1_macro": avg_val_f1_macro,  # Added F1
                }
            )
        if val_attention_maps_to_log:  # Log first attention map image
            log_dict["val/attention_map_example"] = val_attention_maps_to_log[0]
        fabric.log_dict(log_dict)

        if fabric.global_rank == 0:  # Update pbar on rank 0
            epoch_postfix_data = {
                "Train Loss": (
                    f"{avg_epoch_train_loss:.3f}"
                    if not np.isnan(avg_epoch_train_loss)
                    else "N/A"
                ),
            }
            if val_dataloader:
                epoch_postfix_data["Val Loss"] = (
                    f"{avg_epoch_val_loss:.3f}"
                    if not np.isnan(avg_epoch_val_loss)
                    else "N/A"
                )
                epoch_postfix_data["Val Acc"] = (  # Exact Match Accuracy
                    f"{avg_val_exact_match_acc*100:.1f}%"
                    if not np.isnan(avg_val_exact_match_acc)
                    else "N/A"
                )
                epoch_postfix_data["Val F1"] = (  # F1 Macro
                    f"{avg_val_f1_macro:.3f}"
                    if not np.isnan(avg_val_f1_macro)
                    else "N/A"
                )
            epoch_pbar.set_postfix(epoch_postfix_data)

            # Checkpointing logic (only if val_loss is valid)
            # if (
            #     args.checkpoint_dir
            #     and val_dataloader
            #     and not np.isnan(avg_epoch_val_loss)
            # ):
            #     state = {
            #         "epoch": epoch + 1,
            #         "model_state_dict": model.state_dict(),  # Or fabric.get_module_state_dict(model)
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "val/loss": avg_epoch_val_loss,
            #         "val/token_accuracy": avg_val_token_acc,
            #         "val/em_accuracy": avg_val_exact_match_acc,
            #         "val/f1_macro": avg_val_f1_macro,
            #         "args": vars(args),
            #         "source_tokenizer_char_to_idx": source_tokenizer.char_to_idx,
            #         "target_tokenizer_char_to_idx": target_tokenizer.char_to_idx,
            #     }
            #     # Top-k checkpointing based on validation loss
            #     if (
            #         len(top_k_checkpoints) < top_k
            #         or avg_epoch_val_loss
            #         < top_k_checkpoints[-1][0]  # Lower loss is better
            #     ):
            #         ckpt_name = f"ckpt_ep{epoch+1}_vl{avg_epoch_val_loss:.3f}_va{avg_val_exact_match_acc:.2f}.pt"
            #         path = os.path.join(args.checkpoint_dir, ckpt_name)
            #         fabric.save(path, state)
            #         fabric.print(
            #             f"Saved top-k checkpoint: {ckpt_name} (Val Loss: {avg_epoch_val_loss:.4f})"
            #         )
            #         top_k_checkpoints.append((avg_epoch_val_loss, epoch + 1, path))
            #         top_k_checkpoints.sort(
            #             key=lambda x: x[0]
            #         )  # Sort by val_loss (ascending)
            #         if len(top_k_checkpoints) > top_k:
            #             removed_ckpt_path = top_k_checkpoints.pop()[
            #                 2
            #             ]  # Remove the worst (highest loss)
            #             if os.path.exists(removed_ckpt_path):
            #                 try:
            #                     os.remove(removed_ckpt_path)
            #                     fabric.print(
            #                         f"Removed old top-k checkpoint: {os.path.basename(removed_ckpt_path)}"
            #                     )
            #                 except OSError as e:
            #                     fabric.print(
            #                         f"Error removing old checkpoint {os.path.basename(removed_ckpt_path)}: {e}"
            #                     )

            if (
                val_dataloader
                and not np.isnan(avg_epoch_val_loss)
                and avg_epoch_val_loss < best_val_loss
            ):
                best_val_loss = avg_epoch_val_loss
                # Optionally save a "best_model.pt" checkpoint here if desired, in addition to top-k
                # path = os.path.join(args.checkpoint_dir, "ckpt_best_val_loss.pt")
                # fabric.save(path, state)
                # fabric.print(f"Saved new best validation loss checkpoint: {os.path.basename(path)} (Loss: {best_val_loss:.4f})")

    if fabric.global_rank == 0 and hasattr(epoch_pbar, "close"):
        epoch_pbar.close()

    # Save last checkpoint
    # if fabric.global_rank == 0 and args.checkpoint_dir:
    #     path = os.path.join(args.checkpoint_dir, "ckpt_last.pt")
    #     # Gather final validation metrics if available, else use last epoch's
    #     final_val_loss = (
    #         avg_epoch_val_loss if "avg_epoch_val_loss" in locals() else float("nan")
    #     )
    #     final_val_token_acc = (
    #         avg_val_token_acc if "avg_val_token_acc" in locals() else float("nan")
    #     )
    #     final_val_em_acc = (
    #         avg_val_exact_match_acc
    #         if "avg_val_exact_match_acc" in locals()
    #         else float("nan")
    #     )

    #     state = {
    #         "epoch": args.num_epochs,  # or epoch + 1
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "val/loss": final_val_loss,
    #         "val/token_accuracy": final_val_token_acc,
    #         "val/em_accuracy": final_val_em_acc,
    #         "args": vars(args),
    #         "source_tokenizer_char_to_idx": source_tokenizer.char_to_idx,
    #         "target_tokenizer_char_to_idx": target_tokenizer.char_to_idx,
    #     }
    #     fabric.save(path, state)
    #     fabric.print(f"Saved last checkpoint: {os.path.basename(path)}")

    # --- Final Evaluation on Test Set (if available and on rank 0) ---
    if fabric.global_rank == 0 and test_dataloader:
        fabric.print("\n--- Final Evaluation on Test Set ---")
        # Optionally load the best checkpoint for final test evaluation
        best_ckpt_path = None
        if top_k_checkpoints:  # If top-k checkpoints were saved
            best_ckpt_path = top_k_checkpoints[0][
                2
            ]  # Path of the best checkpoint (lowest val_loss)
        elif os.path.exists(
            os.path.join(args.checkpoint_dir or "", "ckpt_best_val_loss.pt")
        ):  # Fallback if specific best was saved
            best_ckpt_path = os.path.join(args.checkpoint_dir, "ckpt_best_val_loss.pt")

        if (
            best_ckpt_path and os.path.exists(best_ckpt_path) and args.checkpoint_dir
        ):  # Check if checkpoint_dir is not None
            fabric.print(
                f"Loading best model from: {os.path.basename(best_ckpt_path)} for test evaluation."
            )
            # Note: fabric.load modifies optimizer and model in place if they are passed.
            # Here, we just need to load model state_dict.
            checkpoint = torch.load(
                best_ckpt_path, map_location=fabric.device
            )  # Load onto correct device
            model.load_state_dict(checkpoint["model_state_dict"])
            # If using fabric.load, it's more involved with `fabric.load_raw` or `fabric.load` with state.
        else:
            fabric.print(
                "No best checkpoint found or specified. Evaluating with the final model state from training."
            )

        test_loss, test_token_acc, test_exact_match_acc, test_f1_macro = (
            evaluate_on_dataloader(
                fabric,
                model,  # Model is already setup with fabric
                test_dataloader,  # Dataloader is already setup
                criterion,
                target_tokenizer,
                set_name="Test",
            )
        )

        test_metrics_str = (
            f"  Test Loss: {test_loss:.4f}\n"
            f"  Test Token Acc: {test_token_acc*100:.2f}%\n"
            f"  Test Exact Match: {test_exact_match_acc*100:.2f}%\n"
            f"  Test F1 Macro: {test_f1_macro:.4f}"
        )
        fabric.print(test_metrics_str)

        
        fabric.log_dict(
            {
                "test/loss": test_loss,
                "test/token_accuracy": test_token_acc,
                "test/exact_match_accuracy": test_exact_match_acc,
                "test/f1_macro": test_f1_macro,
            },
            step=args.num_epochs + 1,  # Log as a final step
        )
        run = fabric.logger.experiment
        run.summary["best_val_loss"] = (
                best_val_loss
                if best_val_loss != float("inf")
                else (
                    avg_epoch_val_loss
                    if "avg_epoch_val_loss" in locals()
                    and not np.isnan(avg_epoch_val_loss)
                    else None
                )
            )
        run.summary["final_test_loss"] = test_loss
        run.summary["final_test_token_accuracy"] = test_token_acc
        run.summary["final_test_exact_match_accuracy"] = test_exact_match_acc
        run.summary["final_test_f1_macro"] = test_f1_macro

    # --- Example Predictions (Post-Training on rank 0) ---
    # example_data_source = None
    # example_data_target = None
    # example_set_name = None

    # if test_source_sents and test_target_sents:
    #     example_data_source, example_data_target, example_set_name = (
    #         test_source_sents,
    #         test_target_sents,
    #         "Test",
    #     )
    # elif dev_source_sents and dev_target_sents:
    #     example_data_source, example_data_target, example_set_name = (
    #         dev_source_sents,
    #         dev_target_sents,
    #         "Dev",
    #     )

    # if (
    #     example_data_source
    #     and example_data_target
    #     and example_set_name
    # ):
    #     fabric.print(
    #         f"\n--- Example Predictions (Post-Training on {example_set_name} Set) ---"
    #     )
    #     table_data = []
    #     model.eval()  # Ensure model is in eval mode

    #     for i in range(min(args.num_prediction_examples, len(example_data_source))):
    #         src_txt, trg_txt = example_data_source[i], example_data_target[i]

    #         greedy_p, _ = (
    #             predict_single_wrapper(  # No attention from greedy in this wrapper version
    #                 model,
    #                 src_txt,
    #                 source_tokenizer,
    #                 target_tokenizer,
    #                 fabric.device,
    #                 beam_size=1,  # Greedy
    #                 max_len=args.inference_max_len,
    #             )
    #         )

    #         beam_p, beam_attn = (
    #             ("N/A", None)  # Default if beam_size <=1
    #             if args.beam_size <= 1
    #             else predict_single_wrapper(
    #                 model,
    #                 src_txt,
    #                 source_tokenizer,
    #                 target_tokenizer,
    #                 fabric.device,
    #                 args.beam_size,
    #                 args.inference_max_len,
    #                 args.length_penalty_alpha,
    #             )
    #         )
    #         table_data.append([src_txt, trg_txt, greedy_p, beam_p])

    #         if beam_attn is not None and args.attention_type.lower() == "bahdanau":
    #             # Tokenize source for heatmap labels
    #             src_tok_ids = source_tokenizer.encode(
    #                 src_txt, add_sos=False, add_eos=False
    #             )  # No special tokens for labels
    #             src_toks_for_map = [
    #                 source_tokenizer.idx_to_char.get(id, "?")
    #                 for id in src_tok_ids
    #                 if id != source_tokenizer.PAD_IDX
    #             ]

    #             # Tokenize beam prediction for heatmap labels
    #             # beam_p is already a string, no need to re-decode IDs unless you need precise tokens before joining
    #             beam_pred_toks_for_map = [char for char in beam_p]  # Simple char list

    #             if (
    #                 src_toks_for_map
    #                 and beam_pred_toks_for_map
    #                 and beam_attn.numel() > 0  # beam_attn is (pred_len, src_len)
    #             ):
    #                 # Ensure attention map dimensions match token list lengths
    #                 # Beam search might produce different pred_len than beam_attn.shape[0] if EOS handling differs
    #                 # Use min(len(beam_pred_toks_for_map), beam_attn.shape[0]) for pred_len
    #                 # Use min(len(src_toks_for_map), beam_attn.shape[1]) for src_len

    #                 attn_pred_len = min(len(beam_pred_toks_for_map), beam_attn.shape[0])
    #                 attn_src_len = min(len(src_toks_for_map), beam_attn.shape[1])

    #                 if attn_pred_len > 0 and attn_src_len > 0:
    #                     attn_plt = beam_attn.cpu().numpy()[
    #                         :attn_pred_len, :attn_src_len
    #                     ]
    #                     current_src_toks = src_toks_for_map[:attn_src_len]
    #                     current_beam_pred_toks = beam_pred_toks_for_map[:attn_pred_len]

    #                     if (
    #                         current_src_toks
    #                         and current_beam_pred_toks
    #                         and attn_plt.size > 0
    #                     ):
    #                         img = plot_attention_heatmap(
    #                             current_src_toks,
    #                             current_beam_pred_toks,
    #                             attn_plt,
    #                             f"BeamAttn ({example_set_name} ex {i+1}: '{src_txt[:20]}...')",
    #                         )
    #                         if img:
    #                             fabric.logger.log_image(key=f"pred_examples/ex{i+1}_beam_attention", images=[wandb.Image(img)], commit=False)  # commit with table

    #     fabric.logger.log_table(key="pred_examples/predictions_table", columns=["Source", "True Target", "Predicted (Greedy)", "Predicted (Beam)"], data=table_data)  # Commit table and any pending images

    return best_val_loss if not np.isnan(best_val_loss) else float("inf")


def predict_single_wrapper(
    model: Seq2Seq,
    source_text: str,
    source_tokenizer: CharTokenizer,
    target_tokenizer: CharTokenizer,
    device: torch.device,  # Explicitly pass device
    beam_size: int = 1,
    max_len: int = 50,
    length_penalty_alpha: float = 0.75,
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    Generates a transliteration for a single source text using greedy or beam search.
    Model should already be on the correct device.
    """
    model.eval()  # Ensure model is in eval mode
    source_ids = source_tokenizer.encode(source_text, add_sos=True, add_eos=True)
    source_tensor = (
        torch.tensor(source_ids, dtype=torch.long).unsqueeze(1).to(device)
    )  # (seq_len, 1)
    source_length = torch.tensor([len(source_ids)], dtype=torch.long).to(
        "cpu"
    )  # Must be on CPU

    predicted_ids: List[int] = []
    attention_matrix: Optional[torch.Tensor] = (
        None  # Expected shape (pred_len, src_len)
    )

    with torch.no_grad():
        if beam_size <= 1:  # Greedy decoding
            # model.forward expects (src, src_len, trg=None, teacher_forcing=0.0, inference_max_len)
            # It returns (decoder_logits, attention_weights)
            # decoder_logits: (max_len, 1, vocab_size)
            # attention_weights: (max_len, 1, src_len) if attention is used
            decoder_logits, attention_weights = model(
                source_tensor,
                source_length,
                None,
                teacher_forcing_ratio=0.0,
                inference_max_len=max_len,
            )
            if decoder_logits.numel() > 0:
                # Get token IDs: (max_len, 1) -> (max_len)
                predicted_ids = decoder_logits.argmax(dim=2).squeeze(1).cpu().tolist()
                if attention_weights is not None and attention_weights.numel() > 0:
                    # Squeeze batch dim: (max_len, src_len)
                    attention_matrix = attention_weights.squeeze(1)
            else:  # Should not happen if max_len > 0
                predicted_ids = [target_tokenizer.EOS_IDX]
        else:  # Beam search decoding
            # beam_search_decode should handle device placement internally or accept device
            predicted_ids, attention_matrix = beam_search_decode(
                model.module,
                source_tensor,  # on device
                source_length,  # on CPU
                target_tokenizer.SOS_IDX,
                target_tokenizer.EOS_IDX,
                beam_size,
                max_len,
                device,  # Pass device to beam_search_decode
                length_penalty_alpha,
            )  # attention_matrix from beam_search_decode should be (pred_len, src_len)

    return (
        target_tokenizer.decode(predicted_ids, remove_special_tokens=True),
        attention_matrix,  # This is (pred_len, src_len)
    )


def get_args_parser() -> argparse.ArgumentParser:
    """
    Defines and returns the ArgumentParser for the script.
    """
    parser = argparse.ArgumentParser(
        description="Seq2Seq Transliteration Training Script with PyTorch Lightning Fabric"
    )
    # Dataset and Paths
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Dakshina dataset directory (e.g., /path/to/dakshina_dataset_hi_v1.0).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,  # Default to 0 for main process dataloading, easier debugging.
        help="Number of dataloader workers per process. Recommended: 0 for small datasets/debugging, os.cpu_count() // num_gpus for larger.",
    )
    # Model Architecture
    parser.add_argument(
        "--input_embed_dim",
        type=int,
        default=128,
        help="Input character embedding dimension.",
    )
    parser.add_argument(
        "--target_embed_dim",
        type=int,
        default=128,
        help="Target character embedding dimension.",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="RNN hidden state dimension."
    )
    parser.add_argument(
        "--num_layers",  # Default for both encoder and decoder if specific not set
        type=int,
        default=2,
        help="Default number of RNN layers for encoder and decoder if not specified otherwise.",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=None,  # Uses num_layers if None
        help="Number of encoder RNN layers. Overrides --num_layers for encoder.",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=None,  # Uses num_layers if None
        help="Number of decoder RNN layers. Overrides --num_layers for decoder.",
    )
    parser.add_argument(
        "--rnn_cell_type",
        type=str,
        default="GRU",
        choices=["RNN", "LSTM", "GRU"],
        help="Type of RNN cell.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability in RNNs and embeddings.",
    )
    parser.add_argument(
        "--encoder_bidirectional",
        action=argparse.BooleanOptionalAction,  # Allows --encoder_bidirectional / --no-encoder_bidirectional
        default=True,
        help="Use bidirectional encoder.",
    )
    # Attention
    parser.add_argument(
        "--attention_type",
        type=str,
        default="none",  # Default to no attention for simpler base model
        choices=["bahdanau", "none"],
        help="Attention mechanism type ('none' or 'bahdanau').",
    )
    parser.add_argument(
        "--attention_dim",  # Relevant only if attention_type is not 'none'
        type=int,
        default=128,
        help="Attention MLP dimension (e.g., for Bahdanau attention).",
    )
    # Training Hyperparameters
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size per device."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--teacher_forcing_ratio",
        type=float,
        default=0.5,
        help="Probability of using teacher forcing during training.",
    )
    parser.add_argument(
        "--grad_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 or negative to disable).",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for CrossEntropyLoss (0.0 for no smoothing).",
    )
    # Inference & Logging
    # parser.add_argument(
    #     "--beam_size",
    #     type=int,
    #     default=1,
    #     help="Beam size for decoding (1 for greedy decoding).",
    # )
    # parser.add_argument(
    #     "--length_penalty_alpha",
    #     type=float,
    #     default=0.75,
    #     help="Length penalty alpha for beam search. (e.g., 0.75). Effective if beam_size > 1.",
    # )
    parser.add_argument(
        "--inference_max_len",
        type=int,
        default=75,  # Max length of generated sequence during inference
        help="Maximum length for generated sequences during inference and prediction examples.",
    )
    parser.add_argument(
        "--log_attention_every_n_epochs",
        type=int,
        default=5,  # Log less frequently by default
        help="Frequency (in epochs) to log attention map examples during validation.",
    )
    parser.add_argument(
        "--num_prediction_examples",
        type=int,
        default=5,
        help="Number of prediction examples to log to W&B table at the end of training.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",  # Default to a local checkpoints directory
        help="Directory to save model checkpoints. If None, no checkpoints are saved.",
    )
    # W&B and Reproducibility
    parser.add_argument(
        "--project_name",
        type=str,
        default="da6401-assignment3",  # Generic project name
        help="W&B project name.",
    )
    parser.add_argument(
        "--run_name",  # Changed from sweep_name for clarity when not sweeping
        type=str,
        default=None,
        help="Custom W&B run name (for manual runs). If None, a name is auto-generated.",
    )
    parser.add_argument(
        "--sweep_config_path",
        type=str,
        default=None,
        help="Path to the W&B sweep configuration YAML file. If provided, a sweep will be run.",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Existing W&B sweep ID to attach an agent to. Overrides sweep_config_path if both provided.",
    )
    parser.add_argument(
        "--disable_wandb", action="store_true", help="Disable W&B logging."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    # Fabric arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16-mixed",  # bf16 is good on modern GPUs
        choices=["32", "16-mixed", "bf16-mixed"],  # "32" will map to "32-true"
        help="Mixed precision setting for L.Fabric.",
    )
    return parser


def generate_wandb_run_name(args: argparse.Namespace, is_sweep: bool) -> Optional[str]:
    """
    Generates a descriptive run name for W&B logging.
    Uses args.run_name if provided for a non-sweep run.
    """
    if args.run_name and not is_sweep:
        return args.run_name

    name_parts = []
    try:
        # Try to get language identifier from dataset_path
        dataset_folder_name = os.path.basename(args.dataset_path.strip(os.sep))
        # Example: dakshina_dataset_hi_v1.0 -> hi
        # Example: en-ta -> en-ta
        parts = dataset_folder_name.split("_")
        if len(parts) > 2 and parts[0] == "dakshina" and parts[1] == "dataset":
            lang_id = parts[2]  # e.g., 'hi'
            name_parts.append(lang_id)
        elif len(parts) == 1 and "-" in parts[0]:  # e.g. 'en-ta'
            name_parts.append(parts[0])
        # else, don't add lang_id if unclear
    except Exception:  # pylint: disable=broad-except
        pass  # Silently ignore if path parsing fails

    name_parts.extend(
        [
            f"{args.rnn_cell_type.lower()}",
            f"h{args.hidden_dim}",
            f"l{(args.num_encoder_layers if args.num_encoder_layers is not None else args.num_layers)}",  # Effective num_layers
            "bidi" if args.encoder_bidirectional else "uni",
            (
                f"attn-{args.attention_type[:3].lower()}{args.attention_dim}"
                if args.attention_type.lower() != "none"
                else "noattn"
            ),
            f"lr{args.learning_rate:.0e}",  # Scientific notation for LR
            f"bs{args.batch_size}",
            f"do{args.dropout:.1f}",  # Dropout with one decimal
        ]
    )
    if not is_sweep:  # Add timestamp for non-sweep runs to ensure uniqueness
        name_parts.append(time.strftime("%m%d-%H%M%S"))

    return "_".join(name_parts)


def run_sweep_agent_function(base_args: argparse.Namespace):
    """
    Function to be called by wandb.agent.
    Initializes W&B for the sweep run, merges config, and launches training.
    """
    # wandb.init() is called by the agent implicitly, or we can call it here
    # to get the sweep's config for this run.
    # For safety, init if not already initialized by agent.
    wandb.init()  # Agent should handle project, sweep_id
    logger = WandbLogger(log_model="all")

    fabric = L.Fabric(
        accelerator=accelerator_type,
        devices=devices_type,
        strategy=strategy,
        precision=fabric_precision,
        loggers=[logger],
    )

    # args is the single source of truth for hyperparameters during the run
    # Copy base_args (from script invocation) and update with sweep parameters
    args_dict_from_script = vars(base_args).copy()

    # wandb.config contains parameters for this specific sweep run
    args_from_sweep = dict(wandb.config)

    # Merge: sweep parameters override script defaults/args
    final_args_dict = {**args_dict_from_script, **args_from_sweep}

    # Construct run name for this sweep trial
    # Convert final_args_dict back to Namespace to pass to generate_wandb_run_name
    temp_args_for_name = argparse.Namespace(**final_args_dict)
    run_name_for_sweep_trial = generate_wandb_run_name(
        temp_args_for_name, is_sweep=True
    )

    # Update W&B run name if possible (agent might control this)
    if run_name_for_sweep_trial and wandb.run:
        try:
            wandb.run.name = run_name_for_sweep_trial
        except Exception as e:  # pylint: disable=broad-except
            fabric.print(f"Warning: Could not set W&B run name for sweep trial: {e}")

    # Handle checkpoint_dir: make it unique for each sweep trial
    base_checkpoint_dir = args_dict_from_script.get(
        "checkpoint_dir", "./checkpoints_sweep"
    )
    if (
        base_checkpoint_dir and run_name_for_sweep_trial
    ):  # Ensure base_checkpoint_dir is not None
        final_args_dict["checkpoint_dir"] = os.path.join(
            base_checkpoint_dir, run_name_for_sweep_trial
        )
    elif base_checkpoint_dir:  # Fallback if run_name is None
        final_args_dict["checkpoint_dir"] = os.path.join(
            base_checkpoint_dir, f"trial_{wandb.run.id if wandb.run else 'unknown'}"
        )
    else:  # If original checkpoint_dir was None, keep it None
        final_args_dict["checkpoint_dir"] = None

    current_run_args = argparse.Namespace(**final_args_dict)

    if fabric.global_rank == 0:  # Print effective config for this trial on rank 0
        fabric.print(f"--- Starting W&B Sweep Trial ---")
        fabric.print(
            f"--- Run Name: {wandb.run.name if wandb.run else 'N/A'} (ID: {wandb.run.id if wandb.run else 'N/A'}) ---"
        )
        fabric.print("Effective arguments for this trial:")
        for arg_name, value in sorted(vars(current_run_args).items()):
            fabric.print(f"  {arg_name}: {value}")
        if current_run_args.checkpoint_dir:
            fabric.print(
                f"Checkpoints for this trial will be saved to: {current_run_args.checkpoint_dir}"
            )

    # Launch the main training function with the combined_args
    # fabric.launch is the key call that starts distributed training if configured
    fabric.launch(train_and_evaluate, current_run_args)
    # wandb.finish() will be called after the agent function finishes, or at the end of __main__


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # Determine if this is a W&B sweep run
    is_sweep_run = args.sweep_id is not None or (
        args.sweep_config_path is not None and os.path.exists(args.sweep_config_path)
    )

    # Setup Fabric
    if args.mixed_precision == "32":
        fabric_precision = "32-true"
    elif args.mixed_precision in ["16-mixed", "bf16-mixed"]:
        fabric_precision = args.mixed_precision
    else:  # Should not be reached due to choices in argparse
        fabric_precision = "32-true"

    strategy = "auto"  # Let Fabric choose, e.g., "ddp" if multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        strategy = (
            "ddp_spawn"  # Be more explicit for multi-GPU, "ddp_spawn" can be slower sometimes
        )

    accelerator_type = "cuda" if torch.cuda.is_available() else "cpu"
    devices_type = "auto"  # Let Fabric manage number of devices (e.g., all GPUs if accelerator is "cuda")

    # fabric = L.Fabric(
    #     accelerator=accelerator_type,
    #     devices=devices_type,
    #     strategy=strategy,
    #     precision=fabric_precision,
    # )
    # fabric.launch will handle calling the main logic on all processes.
    # We need to ensure that wandb.init and agent setup happens *before* fabric.launch
    # for sweep scenarios, or that fabric.launch calls a function that then does this on rank 0.

    # The typical pattern:
    # 1. All processes initialize Fabric.
    # 2. On global_rank == 0: Set up W&B, start sweep agent if applicable.
    # 3. Call fabric.launch(fn, ...). `fn` is your main training logic.
    #    If it's a sweep, `fn` might be a wrapper that gets sweep config and calls train_and_evaluate.
    #    If not a sweep, `fn` is train_and_evaluate directly.

    if is_sweep_run:
        
        sweep_id_to_use = args.sweep_id
        if not sweep_id_to_use:  # Create a new sweep if ID not given
            try:
                with open(args.sweep_config_path, "r", encoding="utf-8") as f:
                    sweep_config_yaml = yaml.safe_load(f)
                # Update project name in sweep_config if not already set from args
                if "project" not in sweep_config_yaml:
                    sweep_config_yaml["project"] = args.project_name
                elif (
                    args.project_name
                    and sweep_config_yaml["project"] != args.project_name
                ):
                    print(
                        f"Warning: Project name in sweep_config ('{sweep_config_yaml['project']}') differs from --project_name ('{args.project_name}'). Using value from sweep_config."
                    )

                sweep_id_to_use = wandb.sweep(
                    sweep_config_yaml,
                    project=sweep_config_yaml.get("project", args.project_name),
                )
                print(f"Started new W&B sweep with ID: {sweep_id_to_use}")
            except Exception as e:  # pylint: disable=broad-except
                print(
                    f"Error creating W&B sweep from config {args.sweep_config_path}: {e}"
                )
                exit(1)

            print(f"Attaching W&B agent to sweep ID: {sweep_id_to_use}")
            # The agent will call `run_sweep_agent_function` for each trial.
            # `run_sweep_agent_function` itself calls `fabric.launch`.
            wandb.agent(
                sweep_id=sweep_id_to_use,
                function=lambda: run_sweep_agent_function(
                    args
                ),  # Pass current fabric and base args
                project=args.project_name,  # Required by agent if not in env
                count=100,  # Max number of runs for this agent, make configurable if needed
            )
        # Other ranks (if any, before fabric.launch inside run_sweep_agent_function) do nothing for sweep setup.
        # They will be activated by fabric.launch from rank 0's agent call.

    else:  # Single (non-sweep) run
        
        if not args.disable_wandb:
            run_name_to_log = generate_wandb_run_name(args, is_sweep=False)
            wandb.init(
                project=args.project_name, config=vars(args), name=run_name_to_log
            )
            print(f"--- Starting Single W&B Run ---")
            print(
                f"--- Run Name: {wandb.run.name if wandb.run else 'N/A'} (ID: {wandb.run.id if wandb.run else 'N/A'}) ---"
            )
        elif args.disable_wandb:
            print("W&B logging is disabled.")

        
        print("Effective arguments for this run:")
        for arg_name, value in sorted(vars(args).items()):
            print(f"  {arg_name}: {value}")
        if args.checkpoint_dir:
            print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

        logger = WandbLogger()
        fabric = L.Fabric(
            accelerator=accelerator_type,
            devices=devices_type,
            strategy=strategy,
            precision=fabric_precision,
            loggers=[logger],
        )

        # All processes call fabric.launch for a single run.
        # Fabric ensures `train_and_evaluate` is executed in the distributed context.
        fabric.launch(train_and_evaluate, args)