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
from sklearn.metrics import f1_score, confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm 

from tokenizer import CharTokenizer
from dataset import TransliterationDataset, collate_fn
from model import EncoderRNN, DecoderRNN, Seq2Seq, BahdanauAttention
from utils import plot_attention_heatmap, beam_search_decode, log_focused_attention_to_wandb


def load_dakshina_split(file_path: Optional[str]) -> Tuple[List[str], List[str]]:
    """
    Loads a Dakshina dataset split from a TSV file.
    (Implementation as provided in the problem description)
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
            print(
                f"Warning ({os.path.basename(file_path if file_path else 'None')}): Less than 2 columns found. Returning empty lists."
            )
            return [], []

        df = df.iloc[:, [0, 1]].copy()
        df.columns = ["target", "source"]

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
                f"Warning ({os.path.basename(file_path if file_path else 'None')}): Significant rows dropped. Initial: {initial_rows}, Final: {len(df)}"
            )

        source_sents = df["source"].tolist()
        target_sents = df["target"].tolist()

    except FileNotFoundError:
        print(f"Warning: Data file not found {file_path}. Returning empty lists.")
    except Exception as e:
        print(
            f"Error reading or processing data file {file_path}: {e}. Returning empty lists."
        )
    return source_sents, target_sents


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    num_classes: int,
    set_name: str = "Validation",
) -> Tuple[float, float, float]:
    """
    Calculates token accuracy, exact match accuracy, and macro F1 score.
    (Implementation as provided in the problem description)
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
        if mask[i].any():
            pred_seq = predictions_sliced[i][mask[i]]
            tgt_seq = targets_sliced[i][mask[i]]
            if pred_seq.tolist() == tgt_seq.tolist():
                exact_matches += 1
        elif not predictions_sliced[i][predictions_sliced[i] != pad_idx].any():
            exact_matches += 1
    exact_match_acc = (
        exact_matches / targets_sliced.size(0) if targets_sliced.size(0) > 0 else 0.0
    )

    f1_macro = 0.0
    y_true_flat = targets_sliced[mask].cpu().numpy()
    y_pred_flat = predictions_sliced[mask].cpu().numpy()

    if len(y_true_flat) > 0 and len(y_pred_flat) > 0:
        f1_macro = f1_score(
            y_true_flat,
            y_pred_flat,
            average="macro",
            labels=np.arange(num_classes),
            zero_division=0,
        )
    return token_acc, exact_match_acc, f1_macro


@torch.no_grad()
def evaluate_on_dataloader(
    fabric: L.Fabric,
    model: Seq2Seq,
    dataloader: DataLoader,
    criterion: nn.Module,
    source_tokenizer: CharTokenizer,
    target_tokenizer: CharTokenizer,
    set_name: str = "val",
    inference_max_len: int = 50,
) -> Tuple[float, float, float, float]:
    """
    Evaluates the model on a given dataloader.
    Logs standard attention heatmap and focused attention for the first sample if requested.
    """
    model.eval()
    total_loss_sum, total_token_acc_sum, total_exact_match_acc_sum, total_f1_macro_sum = (
        0.0, 0.0, 0.0, 0.0
    )
    current_batch_idx = 0
    processed_batches_count = 0

    eval_pbar = tqdm(
        dataloader, desc=f"Evaluating on {set_name}", unit="batch",
        dynamic_ncols=True, disable=(fabric.global_rank != 0), leave=False
    )

    for src_seqs, src_lengths, trg_seqs, trg_lengths in eval_pbar:
        decoder_logits, attention_weights = model(
            src_seqs, src_lengths, trg_seqs, teacher_forcing_ratio=0.0,
        )

        output_len = decoder_logits.shape[0]
        preds_for_loss = decoder_logits.reshape(-1, decoder_logits.shape[-1])
        targets_for_loss = trg_seqs[1 : output_len + 1].reshape(-1)
        predicted_tokens_batch = decoder_logits.argmax(dim=2).permute(1, 0)

        current_batch_loss, current_token_acc, current_exact_match_acc, current_f1_macro = \
            float("nan"), 0.0, 0.0, 0.0

        if preds_for_loss.numel() > 0 and targets_for_loss.numel() > 0:
            loss = criterion(preds_for_loss, targets_for_loss)
            current_batch_loss = loss.item()
            if not np.isnan(current_batch_loss): total_loss_sum += current_batch_loss

            targets_for_metrics = trg_seqs[1 : output_len + 1].permute(1, 0)
            current_token_acc, current_exact_match_acc, current_f1_macro = calculate_metrics(
                predicted_tokens_batch, targets_for_metrics, target_tokenizer.PAD_IDX,
                target_tokenizer.vocab_size, set_name=set_name,
            )
            total_token_acc_sum += current_token_acc
            total_exact_match_acc_sum += current_exact_match_acc
            total_f1_macro_sum += current_f1_macro
        
        # --- Standard Attention Heatmap Logging (for the first sample) ---
        # Log attention for the first sample of the first batch
        if attention_weights is not None and attention_weights.numel() > 0 and \
        fabric.global_rank == 0 and wandb.run is not None:
            try:
                # Ensure there's at least one sample and attention weights have expected dims
                if src_seqs.size(1) > 0 and attention_weights.dim() == 3:
                    # Data for the first sample in the batch
                    first_sample_src_ids = src_seqs[:, 0]
                    first_sample_src_len = src_lengths[0].item()
                    first_sample_pred_ids = predicted_tokens_batch[0] # (pred_len)

                    # Decode source tokens
                    src_tokens_list_ex = source_tokenizer.decode(
                        first_sample_src_ids[:first_sample_src_len].cpu().tolist(),
                        remove_special_tokens=True,
                    )
                    # Convert string to list of chars if plot_attention_heatmap expects list
                    if isinstance(src_tokens_list_ex, str):
                        src_tokens_list_ex = list(src_tokens_list_ex)


                    # Decode predicted target tokens
                    pred_tokens_list_ex = target_tokenizer.decode(
                        first_sample_pred_ids.cpu().tolist(),
                        remove_special_tokens=True,
                        )
                    if isinstance(pred_tokens_list_ex, str):
                        pred_tokens_list_ex = list(pred_tokens_list_ex)
                    
                    # If pred_tokens_list_ex is empty after decoding (e.g., only EOS predicted)
                    if not pred_tokens_list_ex and first_sample_pred_ids.numel() > 0:
                        if first_sample_pred_ids[0].item() == target_tokenizer.EOS_IDX:
                            pred_tokens_list_ex = ['<EOS>'] # Represent EOS if it's the only thing
                        else:
                            pred_tokens_list_ex = ['<UNK>'] # Placeholder for unknown empty prediction


                    # Attention matrix for the first sample
                    # attention_weights: (num_decoding_steps, batch_size, source_len)
                    # We need: (num_predicted_tokens_for_sample, actual_source_len_for_sample)
                    
                    num_actual_predicted_tokens = len(pred_tokens_list_ex)
                    if num_actual_predicted_tokens == 0 and output_len > 0 : # Predicted empty string, but model ran for output_len steps
                        # This can happen if all predicted tokens were filtered out (e.g. only PAD/SOS)
                        # Or if decode logic results in empty due to early EOS and no EOS char added.
                        # Let's use a placeholder for y-axis if pred_tokens_list_ex is empty
                        # but attention matrix has decoding steps.
                        # However, plot_attention_heatmap might fail if y-labels are empty and matrix isn't.
                        # It's better if pred_tokens_list_ex always has at least one element if num_actual_predicted_tokens>0
                        # For now, if pred_tokens_list_ex is truly empty, we might skip plotting or handle it in plot_attention_heatmap
                        pass


                    # Slice attention matrix:
                    # 1. Select for the first sample in batch: attention_weights[:, 0, ...]
                    # 2. Slice by actual source length: attention_weights[:, 0, :first_sample_src_len]
                    # 3. Slice by number of actual predicted tokens: attention_weights[:num_actual_predicted_tokens, 0, :first_sample_src_len]
                    single_attn_matrix_np = attention_weights[
                        :num_actual_predicted_tokens, # Use length of decoded prediction y-labels
                        0,                             # First sample in batch
                        :first_sample_src_len         # Actual length of source
                    ].cpu().numpy()

                    if src_tokens_list_ex and pred_tokens_list_ex and single_attn_matrix_np.size > 0:
                        # Ensure dimensions match for plotting
                        # single_attn_matrix_np should be (len(pred_tokens_list_ex), len(src_tokens_list_ex))
                        # The slicing above should already handle pred_len. Source len is also handled.
                        # One final check: plot_attention_heatmap might expect src_tokens_list_ex to match matrix's second dim
                        # This should be true due to :first_sample_src_len slicing of attention and src_ids
                        
                        attn_map_img_pil = plot_attention_heatmap(
                            src_tokens_list_ex,
                            pred_tokens_list_ex,
                            single_attn_matrix_np,
                            title=f"{set_name} Attention (Batch 0, Sample 0)",
                        )
                        if attn_map_img_pil:
                            fabric.log_dict({f"{set_name}/attention_first_sample": attn_map_img_pil})
                            fabric.print(f"Logged attention map for {set_name} (first sample) to W&B.")
                    else:
                        fabric.print(f"Skipped logging attention for {set_name}: Empty tokens or attention matrix after processing for heatmap.")

                else: # Batch size is 0 or attention_weights has unexpected dimensions
                    fabric.print(f"Skipped logging attention for {set_name}: Batch size is 0 or attention_weights dimensions are unexpected.")
            except Exception as e:
                fabric.print(f"Error during {set_name} attention visualization: {e}")
                import traceback
                traceback.print_exc()


        # --- Focused Attention Interpretability Plot Logging (for the first sample) ---
        if set_name == "test" and attention_weights is not None and attention_weights.numel() > 0 and \
           fabric.global_rank == 0 and wandb.run is not None:
            try:
                if src_seqs.size(1) > 0 and attention_weights.dim() == 3 and predicted_tokens_batch.size(0) > 0:
                    # Data for the first sample in the batch
                    first_sample_src_ids = src_seqs[:, 0][:src_lengths[0].item()]
                    first_sample_pred_ids_with_eos = predicted_tokens_batch[0] # Includes potential EOS

                    source_text = source_tokenizer.decode(
                        first_sample_src_ids.cpu().tolist(),
                        remove_special_tokens=True, # Keep this simple for direct use
                    )
                    
                    # For focused attention, we need predicted tokens up to EOS
                    # The length of this list will determine how many rows of attention to take
                    predicted_text_list_for_attn = target_tokenizer.decode(
                        first_sample_pred_ids_with_eos.cpu().tolist(),
                        remove_special_tokens=True, # Remove SOS, PAD
                        # Return as string, plot_focused_attention_visualization expects string
                    )
                    if not predicted_text_list_for_attn and first_sample_pred_ids_with_eos.numel() > 0:
                         predicted_text_list_for_attn = "<EOS>" # if only EOS was predicted or empty

                    num_actual_predicted_tokens = len(predicted_text_list_for_attn)

                    if num_actual_predicted_tokens > 0 and len(source_text) > 0:
                        # Attention weights: (num_decoding_steps, batch_size, source_len)
                        # We need: (num_actual_predicted_tokens, actual_source_len_for_sample)
                        focused_attn_weights_np = attention_weights[
                            :num_actual_predicted_tokens, # Slice by actual predicted tokens
                            0,                             # First sample in batch
                            :len(source_text)              # Slice by actual source length
                        ].cpu().numpy()
                        
                        log_focused_attention_to_wandb(
                            fabric_logger=fabric.logger, # Pass the fabric logger
                            source_str=source_text,
                            predicted_str=predicted_text_list_for_attn, # Pass the string of predicted chars
                            attention_weights_np=focused_attn_weights_np,
                            log_key_prefix=set_name,
                        )
                    else:
                        fabric.print(f"Skipped focused attention for {set_name}: Empty source or predicted text after decoding.")
            except Exception as e:
                fabric.print(f"Error during {set_name} focused attention visualization: {e}")
                import traceback
                traceback.print_exc()

        processed_batches_count += 1
        current_batch_idx += 1

        if fabric.global_rank == 0:
            eval_pbar.set_postfix({
                f"{set_name} Loss": f"{current_batch_loss:.3f}" if not np.isnan(current_batch_loss) else "N/A",
                "EM Acc": f"{current_exact_match_acc:.3f}", "F1": f"{current_f1_macro:.3f}"
            })

    if hasattr(eval_pbar, 'close'): eval_pbar.close()
    if processed_batches_count == 0: return float("nan"), float("nan"), float("nan"), float("nan")

    avg_loss = total_loss_sum / processed_batches_count
    avg_token_acc = total_token_acc_sum / processed_batches_count
    avg_exact_match_acc = total_exact_match_acc_sum / processed_batches_count
    avg_f1_macro = total_f1_macro_sum / processed_batches_count
    return avg_loss, avg_token_acc, avg_exact_match_acc, avg_f1_macro


def train_and_evaluate(fabric: L.Fabric, args: argparse.Namespace):
    """
    Main training and evaluation process.
    (Modified extensively for requested features)
    """
    fabric.seed_everything(args.seed)

    # --- 1. Load Data ---
    if not args.dataset_path or not os.path.isdir(args.dataset_path):
        message = f"Error: Dataset path '{args.dataset_path}' is not a valid directory."
        fabric.print(message); raise FileNotFoundError(message)

    all_files_in_path = os.listdir(args.dataset_path)
    lang_code_from_path = ""
    try:
        parts = os.path.basename(args.dataset_path.strip(os.sep)).split("_")
        if len(parts) > 2 and parts[0] == "dakshina" and parts[1] == "dataset" and len(parts[2]) == 2: 
            lang_code_from_path = parts[2]
    except Exception: pass 

    def find_file(suffix):
        if lang_code_from_path:
            specific_name = f"{lang_code_from_path}_{suffix}.tsv"
            if specific_name in all_files_in_path: return specific_name
        found_file = next((f for f in all_files_in_path if f.endswith(f"{suffix}.tsv")), None)
        if found_file: return found_file
        if lang_code_from_path:
            found_file = next((f for f in all_files_in_path if f == f"{lang_code_from_path}.{suffix}.tsv"), None)
            if found_file: return found_file
        return next((f for f in all_files_in_path if f.endswith(f".{suffix}.tsv")), None)

    train_file = os.path.join(args.dataset_path, fn) if (fn := find_file("train")) else None
    dev_file = os.path.join(args.dataset_path, fn) if (fn := find_file("dev")) else None
    test_file = os.path.join(args.dataset_path, fn) if (fn := find_file("test")) else None
    
    fabric.print(f"Training data: {train_file or 'Not found'}")
    train_source_sents, train_target_sents = load_dakshina_split(train_file)
    fabric.print(f"Validation data: {dev_file or 'Not found'}")
    dev_source_sents, dev_target_sents = load_dakshina_split(dev_file)
    fabric.print(f"Test data: {test_file or 'Not found'}")
    test_source_sents, test_target_sents = load_dakshina_split(test_file)

    if not train_source_sents:
        message = f"Fatal: No training data from '{train_file}'."; fabric.print(message); raise ValueError(message)

    # --- 2. Tokenizers ---
    source_tokenizer = CharTokenizer()
    source_tokenizer.fit(train_source_sents)
    target_tokenizer = CharTokenizer()
    target_tokenizer.fit(train_target_sents)

    if fabric.global_rank == 0 and args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        source_tokenizer.save_tokenizer(os.path.join(args.checkpoint_dir, "source_tokenizer.json"))
        target_tokenizer.save_tokenizer(os.path.join(args.checkpoint_dir, "target_tokenizer.json"))
    
    source_vocab_size, target_vocab_size = source_tokenizer.vocab_size, target_tokenizer.vocab_size
    if fabric.global_rank == 0:
        fabric.print(f"Src vocab: {source_vocab_size}, Trg vocab: {target_vocab_size}")
        if wandb.run is not None:
            wandb.config.update({
                "source_vocab_size_runtime": source_vocab_size,
                "target_vocab_size_runtime": target_vocab_size,
            }, allow_val_change=True)

    # --- 3. Datasets and DataLoaders ---
    train_dataset_obj = TransliterationDataset(train_source_sents, train_target_sents, source_tokenizer, target_tokenizer, verbose=(fabric.global_rank == 0))
    val_dataset_obj = None
    if dev_source_sents:
        val_dataset_obj = TransliterationDataset(dev_source_sents, dev_target_sents, source_tokenizer, target_tokenizer, verbose=(fabric.global_rank == 0))
    elif dev_file and not dev_source_sents: fabric.print(f"Warning: Dev file {dev_file} loaded no data.")

    if val_dataset_obj is None and len(train_dataset_obj) >= 10:
        fabric.print("No val data, splitting from train.")
        val_len = max(1, int(0.1 * len(train_dataset_obj)))
        train_len = len(train_dataset_obj) - val_len
        if val_len > 0 and train_len > 0:
            train_dataset_obj, val_dataset_obj = torch.utils.data.random_split(
                train_dataset_obj, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
            fabric.print(f"Created val split: {len(train_dataset_obj)} train, {len(val_dataset_obj)} val.")
        else: fabric.print("Could not create val split."); val_dataset_obj = None
    elif val_dataset_obj is None: fabric.print("No val data and train too small for split. Val skipped.")

    collate_fn_partial = partial(collate_fn, source_pad_idx=source_tokenizer.PAD_IDX, target_pad_idx=target_tokenizer.PAD_IDX)
    num_workers = min(max(0, args.num_workers), (os.cpu_count() or 1) // fabric.world_size)
    if fabric.global_rank == 0: fabric.print(f"Using {num_workers} dataloader workers per process.")

    train_dataloader = DataLoader(train_dataset_obj, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_partial, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=(fabric.world_size > 1))
    val_dataloader = DataLoader(val_dataset_obj, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_partial, num_workers=num_workers, pin_memory=torch.cuda.is_available()) if val_dataset_obj and len(val_dataset_obj) > 0 else None
    
    test_dataloader = None
    if test_source_sents and test_target_sents:
        test_dataset_obj = TransliterationDataset(test_source_sents, test_target_sents, source_tokenizer, target_tokenizer, verbose=False)
        if len(test_dataset_obj) > 0:
            test_dataloader = DataLoader(test_dataset_obj, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_partial, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        else: fabric.print("Test data empty after processing.")

    train_dataloader = fabric.setup_dataloaders(train_dataloader) # DistributedSampler is True by default
    if val_dataloader: val_dataloader = fabric.setup_dataloaders(val_dataloader, use_distributed_sampler=False)
    if test_dataloader: test_dataloader = fabric.setup_dataloaders(test_dataloader, use_distributed_sampler=False)
    
    # --- 4. Model ---
    enc_layers = args.num_encoder_layers if args.num_encoder_layers is not None else args.num_layers
    dec_layers = args.num_decoder_layers if args.num_decoder_layers is not None else args.num_layers
    encoder = EncoderRNN(source_vocab_size, args.input_embed_dim, args.hidden_dim, enc_layers, args.rnn_cell_type, args.dropout, args.encoder_bidirectional)
    enc_out_dim = args.hidden_dim * (2 if args.encoder_bidirectional else 1)
    attention = BahdanauAttention(enc_out_dim, args.hidden_dim, args.attention_dim) if args.attention_type.lower() == "bahdanau" else None
    decoder = DecoderRNN(target_vocab_size, args.target_embed_dim, args.hidden_dim, dec_layers, args.rnn_cell_type, args.dropout, attention, enc_out_dim if attention else None)
    model = Seq2Seq(encoder, decoder, target_tokenizer.SOS_IDX, target_tokenizer.EOS_IDX, fabric.device)
    model = fabric.setup_module(model)

    if fabric.global_rank == 0 and wandb.run is not None:
        log_freq = 100
        if hasattr(train_dataloader, "__len__") and (dl_len := len(train_dataloader)) > 1: log_freq = min(100, dl_len // 2 if dl_len > 1 else 1)
        try: wandb.watch(model, log="all", log_freq=log_freq)
        except Exception as e: fabric.print(f"Warning: wandb.watch failed: {e}")

    # --- 5. Optimizer and Loss ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, fused=torch.cuda.is_available())
    optimizer = fabric.setup_optimizers(optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=target_tokenizer.PAD_IDX, label_smoothing=args.label_smoothing)

    # --- 6. Training Loop ---
    if fabric.global_rank == 0: fabric.print("Starting training...")
    best_val_loss = float("inf")
    top_k_checkpoints: List[Tuple[float, int, str]] = []
    top_k = 3

    epoch_pbar = tqdm(range(args.num_epochs), desc="Epochs", unit="epoch", disable=(fabric.global_rank != 0), dynamic_ncols=True)
    if fabric.global_rank == 0:
        pfix = {"TrL": "N/A"}; 
        if val_dataloader: pfix.update({"VL": "N/A", "VAcc": "N/A", "VF1": "N/A"})
        epoch_pbar.set_postfix(pfix)

    for epoch in epoch_pbar:
        model.train()
        epoch_train_loss_sum, num_train_batches, start_time = 0.0, 0, time.time()
        avg_epoch_train_loss = float("nan")

        if not hasattr(train_dataloader, "__len__") or len(train_dataloader) == 0:
            if fabric.global_rank == 0: tqdm.write(f"Epoch {epoch+1}: Train DL empty/unsized. Skipping.")
        else:
            train_batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", total=len(train_dataloader) if hasattr(train_dataloader, "__len__") else None,
                                    disable=(fabric.global_rank != 0), leave=False, unit="batch", dynamic_ncols=True)
            if fabric.global_rank == 0: train_batch_pbar.set_postfix(train_loss="N/A")

            for src_seqs, src_lengths, trg_seqs, trg_lengths in train_batch_pbar:
                optimizer.zero_grad()
                decoder_logits, _ = model(src_seqs, src_lengths, trg_seqs, args.teacher_forcing_ratio)
                out_len = decoder_logits.shape[0]
                preds_loss = decoder_logits.reshape(-1, decoder_logits.shape[-1])
                tgts_loss = trg_seqs[1:out_len+1].reshape(-1)

                if preds_loss.numel() == 0 or tgts_loss.numel() == 0:
                    if fabric.global_rank == 0: train_batch_pbar.set_postfix(train_loss="N/A (empty_batch)")
                    continue
                
                loss = criterion(preds_loss, tgts_loss)
                fabric.backward(loss)
                if args.grad_clip_val > 0: fabric.clip_gradients(model, optimizer, clip_val=args.grad_clip_val)
                optimizer.step()

                batch_loss = loss.item()
                if not np.isnan(batch_loss): epoch_train_loss_sum += batch_loss
                num_train_batches += 1
                fabric.log_dict({"train/batch_loss": batch_loss})
                if fabric.global_rank == 0: train_batch_pbar.set_postfix(train_loss=f"{batch_loss:.3f}" if not np.isnan(batch_loss) else "N/A")
            
            avg_epoch_train_loss = epoch_train_loss_sum / num_train_batches if num_train_batches > 0 else float("nan")
            if fabric.global_rank == 0 and hasattr(train_batch_pbar, "close"): train_batch_pbar.close()
        
        epoch_duration = time.time() - start_time
        avg_val_loss, avg_val_tok_acc, avg_val_em_acc, avg_val_f1 = float("nan"), float("nan"), float("nan"), float("nan")
        val_attn_maps_log = []

        if fabric.global_rank == 0: # Validation only on rank 0
            if val_dataloader and (len(val_dataloader) > 0 if hasattr(val_dataloader, "__len__") else True):
                avg_val_loss, avg_val_tok_acc, avg_val_em_acc, avg_val_f1 = evaluate_on_dataloader(
                    fabric, model, val_dataloader, criterion, source_tokenizer, target_tokenizer, "Validation", args.inference_max_len)

                if args.attention_type.lower() == "bahdanau" and (epoch + 1) % args.log_attention_every_n_epochs == 0 and wandb.run is not None:
                    try:
                        src_s_attn, src_l_attn, trg_s_attn, _ = next(iter(val_dataloader))
                        _, attn_weights_viz = model(src_s_attn.to(fabric.device), src_l_attn, None, 0.0, inference_max_len=trg_s_attn.size(0)-1)
                        if attn_weights_viz is not None and attn_weights_viz.numel() > 0:
                            src_len_ex = src_l_attn[0].item()
                            single_attn = attn_weights_viz[:, 0, :src_len_ex].cpu().numpy()
                            src_ids_ex = src_s_attn[:src_len_ex, 0].cpu().tolist()
                            src_toks_ex = [source_tokenizer.idx_to_char.get(i, "?") for i in src_ids_ex if i != source_tokenizer.PAD_IDX]
                            
                            val_logits_viz, _ = model(src_s_attn.to(fabric.device), src_l_attn, None, 0.0, inference_max_len=trg_s_attn.size(0)-1)
                            if val_logits_viz.numel() > 0:
                                pred_ids_hm_ex = val_logits_viz.argmax(dim=2)[:, 0].cpu().tolist()
                                pred_toks_ex = [target_tokenizer.idx_to_char.get(i, "?") for i in pred_ids_hm_ex if i not in [target_tokenizer.PAD_IDX, target_tokenizer.SOS_IDX, target_tokenizer.EOS_IDX] and i < target_tokenizer.vocab_size]
                                if len(pred_toks_ex) == 0 and len(single_attn) > 0 : pred_toks_ex = ["<EOS>"] # handle empty pred
                                single_attn = single_attn[:len(pred_toks_ex), :] # Align attention map with actual prediction length
                                
                                if src_toks_ex and pred_toks_ex and single_attn.size > 0:
                                    img_pil = plot_attention_heatmap(src_toks_ex, pred_toks_ex, single_attn, f"Ep {epoch+1} Val Attn")
                                    if img_pil: val_attn_maps_log.append(wandb.Image(img_pil))
                    except StopIteration: fabric.print("Warn: Val DL empty for attn viz.")
                    except Exception as e: fabric.print(f"Error in attn viz: {e}")
        
        log_dict_ep: Dict[str, Any] = {"epoch": epoch + 1, "epoch_duration_secs": epoch_duration, "train/epoch_loss": avg_epoch_train_loss, "learning_rate": optimizer.param_groups[0]["lr"]}
        if val_dataloader: log_dict_ep.update({"val/loss": avg_val_loss, "val/token_accuracy": avg_val_tok_acc, "val/acc": avg_val_em_acc, "val/f1_macro": avg_val_f1})
        if val_attn_maps_log and wandb.run is not None: log_dict_ep["val/attention_map_example"] = val_attn_maps_log[0]
        if wandb.run is not None: fabric.log_dict(log_dict_ep)

        if fabric.global_rank == 0:
            pfix_data = {"TrL": f"{avg_epoch_train_loss:.3f}" if not np.isnan(avg_epoch_train_loss) else "N/A"}
            if val_dataloader: pfix_data.update({"VL": f"{avg_val_loss:.3f}" if not np.isnan(avg_val_loss) else "N/A",
                                                 "VAcc": f"{avg_val_em_acc*100:.1f}%" if not np.isnan(avg_val_em_acc) else "N/A",
                                                 "VF1": f"{avg_val_f1:.3f}" if not np.isnan(avg_val_f1) else "N/A"})
            epoch_pbar.set_postfix(pfix_data)

            if args.checkpoint_dir and val_dataloader and not np.isnan(avg_val_loss):
                state = {"epoch": epoch + 1, "model_state_dict": model.module.state_dict(),
                         "val/loss": avg_val_loss, "args": vars(args), 
                         "source_tokenizer_char_to_idx": source_tokenizer.char_to_idx, "target_tokenizer_char_to_idx": target_tokenizer.char_to_idx}
                if len(top_k_checkpoints) < top_k or avg_val_loss < top_k_checkpoints[-1][0]:
                    ckpt_name = f"ckpt_ep{epoch+1}_vl{avg_val_loss:.3f}.pt"
                    path = os.path.join(args.checkpoint_dir, ckpt_name)
                    torch.save(state, path)
                    top_k_checkpoints.append((avg_val_loss, epoch + 1, path))
                    top_k_checkpoints.sort(key=lambda x: x[0])
                    if len(top_k_checkpoints) > top_k:
                        if os.path.exists(rm_path := top_k_checkpoints.pop()[2]):
                            try: os.remove(rm_path)
                            except OSError as e: fabric.print(f"Error removing old ckpt {os.path.basename(rm_path)}: {e}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    path = os.path.join(args.checkpoint_dir, "ckpt_best_val_loss.pt")
                    torch.save(state, path)
                    fabric.print(f"Saved new best val_loss ckpt: {os.path.basename(path)} (Loss: {best_val_loss:.4f})")

    if fabric.global_rank == 0 and hasattr(epoch_pbar, "close"): epoch_pbar.close()

    if fabric.global_rank == 0 and args.checkpoint_dir:
        state = {"epoch": args.num_epochs, "model_state_dict": model.module.state_dict(),
                 "val/loss": avg_val_loss if 'avg_val_loss' in locals() and not np.isnan(avg_val_loss) else None,
                 "args": vars(args), "source_tokenizer_char_to_idx": source_tokenizer.char_to_idx, "target_tokenizer_char_to_idx": target_tokenizer.char_to_idx}
        torch.save(state, os.path.join(args.checkpoint_dir, "ckpt_last.pt"))
        fabric.print(f"Saved last checkpoint.")

    # --- Final Evaluation on Test Set (if available and on rank 0) ---
    if fabric.global_rank == 0 and test_dataloader:
        fabric.print("\n--- Final Evaluation on Test Set ---")
        best_ckpt_load_path = None
        if top_k_checkpoints: best_ckpt_load_path = top_k_checkpoints[0][2]
        elif args.checkpoint_dir and os.path.exists(p := os.path.join(args.checkpoint_dir, "ckpt_best_val_loss.pt")): best_ckpt_load_path = p
        
        if best_ckpt_load_path and os.path.exists(best_ckpt_load_path):
            fabric.print(f"Loading best model from: {os.path.basename(best_ckpt_load_path)} for test eval.")
            checkpoint = torch.load(best_ckpt_load_path, map_location=fabric.device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else: fabric.print("No best ckpt found. Evaluating with final model state.")

        test_loss, test_tok_acc, test_em_acc, test_f1 = evaluate_on_dataloader(
            fabric, model, test_dataloader, criterion, source_tokenizer, target_tokenizer, "Test", args.inference_max_len)
        fabric.print(f"  Test Loss: {test_loss:.4f}\n  Test Token Acc: {test_tok_acc*100:.2f}%\n  Test Exact Match: {test_em_acc*100:.2f}%\n  Test F1 Macro: {test_f1:.4f}")

        test_log_metrics = {"test/loss": test_loss, "test/token_accuracy": test_tok_acc, "test/exact_match_accuracy": test_em_acc, "test/f1_macro": test_f1}
        if wandb.run is not None:
            fabric.log_dict(test_log_metrics, step=args.num_epochs + 1)
            run = fabric.logger.experiment # type: ignore
            run.summary["best_val_loss"] = best_val_loss if best_val_loss != float("inf") else (avg_val_loss if 'avg_val_loss' in locals() and not np.isnan(avg_val_loss) else None)
            run.summary.update({f"final_{k.replace('/', '_')}": v for k,v in test_log_metrics.items()})


        # --- Generate Test Set Predictions CSV, W&B Table, and Confusion Matrix ---
        fabric.print("\n--- Generating Test Set Predictions and Confusion Matrix Details ---")
        model.eval()
        all_true_chars_cm, all_pred_chars_cm = [], []
        sources_csv, targets_csv, preds_csv = [], [], []

        test_detail_pbar = tqdm(test_dataloader, desc="Generating Test Details", unit="batch", dynamic_ncols=True, disable=(fabric.global_rank != 0), leave=False)
        with torch.no_grad():
            for src_s, src_l, trg_s, trg_l in test_detail_pbar:
                logits, _ = model(src_s.to(fabric.device), src_l, None, 0.0, inference_max_len=args.inference_max_len)
                pred_tokens_batch = logits.argmax(dim=2).permute(1, 0) # (B, S_pred)
                
                # For CM: targets need to be aligned (SOS removed, same length as pred or vice-versa)
                true_tokens_batch_cm = trg_s[1:].permute(1,0) # (B, S_true-1)
                max_len_cm = min(pred_tokens_batch.size(1), true_tokens_batch_cm.size(1))
                pred_cm_slice = pred_tokens_batch[:, :max_len_cm]
                true_cm_slice = true_tokens_batch_cm[:, :max_len_cm]
                mask_cm = true_cm_slice != target_tokenizer.PAD_IDX
                all_true_chars_cm.extend(true_cm_slice[mask_cm].cpu().tolist())
                all_pred_chars_cm.extend(pred_cm_slice[mask_cm].cpu().tolist())

                # For CSV/Table: Decode full sequences
                for i in range(src_s.size(1)): # Batch dim
                    src_txt = source_tokenizer.decode(src_s[:src_l[i].item(), i].tolist(), remove_special_tokens=True)
                    trg_txt = target_tokenizer.decode(trg_s[:trg_l[i].item(), i].tolist(), remove_special_tokens=True)
                    pred_txt = target_tokenizer.decode(pred_tokens_batch[i].cpu().tolist(), remove_special_tokens=True)
                    sources_csv.append(src_txt); targets_csv.append(trg_txt); preds_csv.append(pred_txt)
        if hasattr(test_detail_pbar, "close"): test_detail_pbar.close()
        
        if sources_csv:
            pred_df = pd.DataFrame({'source': sources_csv, 'target': targets_csv, 'prediction': preds_csv})
            pred_df.to_csv("test_predictions.csv", index=False, encoding='utf-8')
            fabric.print(f"Saved all test predictions to test_predictions.csv")
            if wandb.run is not None:
                wandb_table = wandb.Table(dataframe=pred_df.head(20))
                fabric.log_dict({"test/predictions_sample": wandb_table}, step=args.num_epochs + 1)
                fabric.print("Uploaded sample of 20 test predictions to W&B.")

        if all_true_chars_cm and all_pred_chars_cm and wandb.run is not None:
            special_tokens = {target_tokenizer.PAD_IDX, target_tokenizer.SOS_IDX, target_tokenizer.EOS_IDX}
            cm_labels_idx = sorted(list(l for l in set(all_true_chars_cm) | set(all_pred_chars_cm) if l not in special_tokens and l < target_tokenizer.vocab_size))
            if not cm_labels_idx: fabric.print("No non-special chars for CM. Skipping.")
            else:
                cm = sk_confusion_matrix(all_true_chars_cm, all_pred_chars_cm, labels=cm_labels_idx, normalize='true') * 100
                cm_char_labels = [target_tokenizer.idx_to_char[i] for i in cm_labels_idx]

                font_path_or_name="TiroDevanagariHindi-Regular.ttf"
                if font_path_or_name and font_path_or_name.lower().endswith(('.ttf', '.otf')) and os.path.exists(font_path_or_name):
                    try:
                        font_props_to_use = fm.FontProperties(fname=font_path_or_name)
                        final_font_description = os.path.basename(font_path_or_name)
                        print(f"Using font file: '{final_font_description}' for all plots.")
                    except Exception as e:
                        print(f"Warning: Could not load specified font file. {e}. Using fallback.")
                        font_props_to_use = fm.FontProperties(family='monospace') # Fallback
                        final_font_description = "monospace (fallback)"
                else: # Fallback if no specific ttf is given or found
                    font_props_to_use = fm.FontProperties(family='monospace')
                    final_font_description = "monospace (default)"
                    print(f"Using default font: {final_font_description}")

                fig_w = max(10, len(cm_char_labels)//2 if len(cm_char_labels) <= 100 else 50)
                fig_h = max(8, len(cm_char_labels)//2.5 if len(cm_char_labels) <= 100 else 40)
                plt.figure(figsize=(fig_w, fig_h))

                # Apply font properties to heatmap annotations
                sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                            xticklabels=cm_char_labels, yticklabels=cm_char_labels, 
                            annot_kws={"size": 8, "fontproperties": font_props_to_use})

                # Apply font properties to labels and title
                plt.xlabel('Predicted Label', fontproperties=font_props_to_use)
                plt.ylabel('True Label', fontproperties=font_props_to_use)
                plt.title('Test Set Character-Level Confusion Matrix', fontproperties=font_props_to_use)

                # Apply font properties to tick labels
                plt.xticks(rotation=45, ha='right', fontproperties=font_props_to_use)
                plt.yticks(rotation=0, fontproperties=font_props_to_use)

                plt.tight_layout()
                try:
                    plt.savefig("confusion_matrix.jpg"); fabric.print(f"Saved CM to confusion_matrix.jpg")
                    fabric.log_dict({"test/confusion_matrix": wandb.Image("confusion_matrix.jpg")}, step=args.num_epochs + 1)
                    fabric.print("Uploaded CM to W&B.")
                except Exception as e: fabric.print(f"Error saving/uploading CM: {e}")
                finally: plt.close()
        elif not (all_true_chars_cm and all_pred_chars_cm): fabric.print("Not enough data for CM.")
    return best_val_loss if not np.isnan(best_val_loss) else float("inf")


def predict_single_wrapper(
    model: Seq2Seq, source_text: str, source_tokenizer: CharTokenizer, target_tokenizer: CharTokenizer,
    device: torch.device, beam_size: int = 1, max_len: int = 50, length_penalty_alpha: float = 0.75
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    Generates transliteration for a single source text.
    (Implementation as provided in the problem description, minor adjustments)
    """
    model.eval()
    source_ids = source_tokenizer.encode(source_text, add_sos=True, add_eos=True)
    source_tensor = torch.tensor(source_ids, dtype=torch.long).unsqueeze(1).to(device)
    source_length = torch.tensor([len(source_ids)], dtype=torch.long).to("cpu") # Lengths must be on CPU for pack_padded

    predicted_ids: List[int] = []
    attention_matrix: Optional[torch.Tensor] = None

    with torch.no_grad():
        unwrapped_model = model.module if hasattr(model, 'module') else model # For beam_search if it expects raw nn.Module
        if beam_size <= 1:
            decoder_logits, attention_weights = unwrapped_model(source_tensor, source_length, None, 0.0, inference_max_len=max_len)
            if decoder_logits.numel() > 0:
                predicted_ids = decoder_logits.argmax(dim=2).squeeze(1).cpu().tolist()
                if attention_weights is not None and attention_weights.numel() > 0:
                    attention_matrix = attention_weights.squeeze(1) 
            else: predicted_ids = [target_tokenizer.EOS_IDX]
        else:
            predicted_ids, attention_matrix = beam_search_decode(
                unwrapped_model, source_tensor, source_length, target_tokenizer.SOS_IDX, target_tokenizer.EOS_IDX,
                beam_size, max_len, device, length_penalty_alpha)
    return target_tokenizer.decode(predicted_ids, remove_special_tokens=True), attention_matrix


def get_args_parser() -> argparse.ArgumentParser:
    """Defines and returns the ArgumentParser."""
    parser = argparse.ArgumentParser(description="Seq2Seq Transliteration Training with L.Fabric")
    # Dataset and Paths
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to Dakshina dataset directory.")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers per process.")
    # Model Architecture
    parser.add_argument("--input_embed_dim", type=int, default=128, help="Input char embedding dim.")
    parser.add_argument("--target_embed_dim", type=int, default=128, help="Target char embedding dim.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN hidden state dim.")
    parser.add_argument("--num_layers", type=int, default=2, help="Default RNN layers for enc/dec.")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Encoder RNN layers (overrides --num_layers).")
    parser.add_argument("--num_decoder_layers", type=int, default=1, help="Decoder RNN layers (overrides --num_layers).")
    parser.add_argument("--rnn_cell_type", type=str, default="LSTM", choices=["RNN", "LSTM", "GRU"], help="RNN cell type.")
    parser.add_argument("--dropout", type=float, default=0.15971115555666263, help="Dropout probability.")
    parser.add_argument("--encoder_bidirectional", action=argparse.BooleanOptionalAction, default=True, help="Use bidirectional encoder.")
    # Attention
    parser.add_argument("--attention_type", type=str, default="bahdanau", choices=["bahdanau", "none"], help="Attention type.")
    parser.add_argument("--attention_dim", type=int, default=128, help="Attention MLP dim (for Bahdanau).")
    # Training Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=0.0015486892210042013, help="Adam learning rate.")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="Teacher forcing probability.")
    parser.add_argument("--grad_clip_val", type=float, default=5.0, help="Gradient clipping value (0 to disable).")
    parser.add_argument("--label_smoothing", type=float, default=0.12425772123435952, help="Label smoothing (0 for none).")
    # Inference & Logging
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for decoding (1=greedy).")
    parser.add_argument("--length_penalty_alpha", type=float, default=0.75, help="Length penalty for beam search.")
    parser.add_argument("--inference_max_len", type=int, default=75, help="Max len for generated sequences.")
    parser.add_argument("--log_attention_every_n_epochs", type=int, default=5, help="Freq (epochs) to log attn maps.")
    parser.add_argument("--num_prediction_examples", type=int, default=5, help="Num W&B table prediction examples (old).")
    parser.add_argument("--checkpoint_dir", type=str, default="./ckpt", help="Dir for model checkpoints.")
    # W&B and Reproducibility
    parser.add_argument("--project_name", type=str, default="da6401-assignment3", help="W&B project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom W&B run name (manual runs).")
    parser.add_argument("--sweep_config_path", type=str, default=None, help="Path to W&B sweep config YAML.")
    parser.add_argument("--sweep_id", type=str, default=None, help="Existing W&B sweep ID to attach agent.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Fabric arguments
    parser.add_argument("--mixed_precision", type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed"], help="Mixed precision for L.Fabric.")
    return parser


def generate_wandb_run_name(args: argparse.Namespace, is_sweep: bool) -> Optional[str]:
    """Generates a descriptive run name for W&B."""
    if args.run_name and not is_sweep: return args.run_name
    name_parts = []
    try:
        folder_name = os.path.basename(args.dataset_path.strip(os.sep))
        p = folder_name.split("_")
        if len(p) > 2 and p[0]=="dakshina" and p[1]=="dataset": name_parts.append(p[2])
        elif len(p)==1 and "-" in p[0]: name_parts.append(p[0])
    except Exception: pass
    name_parts.extend([
        args.rnn_cell_type.lower(), f"h{args.hidden_dim}",
        f"l{(args.num_encoder_layers if args.num_encoder_layers is not None else args.num_layers)}",
        "bidi" if args.encoder_bidirectional else "uni",
        f"attn-{args.attention_type[:3].lower()}{args.attention_dim}" if args.attention_type.lower() != "none" else "noattn",
        f"lr{args.learning_rate:.0e}", f"bs{args.batch_size}", f"do{args.dropout:.1f}"
    ])
    if not is_sweep: name_parts.append(time.strftime("%m%d-%H%M%S"))
    return "_".join(name_parts)


def run_sweep_agent_function(base_args: argparse.Namespace, fabric_config: Dict[str, Any]):
    """
    Function called by wandb.agent for each trial.
    Initializes W&B, Fabric, merges config, and launches training.
    """
    if wandb.run is None: # Agent should have initialized this
        wandb.init() 

    sweep_wandb_logger = WandbLogger(log_model="all") # Or True/False for log_model based on needs
    
    fabric = L.Fabric(
        accelerator=fabric_config["accelerator"],
        devices=fabric_config["devices"],
        strategy=fabric_config["strategy"],
        precision=fabric_config["precision"],
        loggers=[sweep_wandb_logger],
    )

    def train_and_evaluate_sweep_trial_wrapper():
        # This code runs inside each process launched by fabric.launch()
        # `fabric` object is the one configured above and launched.
        
        args_script = vars(base_args).copy()
        args_sweep = dict(wandb.config) # wandb.config has parameters for this specific sweep run
        final_args_dict = {**args_script, **args_sweep}
        
        current_run_args = argparse.Namespace(**final_args_dict)
        run_name_trial = generate_wandb_run_name(current_run_args, is_sweep=True)

        if fabric.global_rank == 0 and run_name_trial and wandb.run:
            try:
                wandb.run.name = run_name_trial
                wandb.run.save() 
            except Exception as e:
                fabric.print(f"Warning: Could not set W&B run name for sweep trial: {e}")

        # Handle checkpoint_dir for sweep trial
        base_ckpt_dir = args_script.get("checkpoint_dir", "./checkpoints_sweep")
        if base_ckpt_dir and run_name_trial:
            current_run_args.checkpoint_dir = os.path.join(base_ckpt_dir, run_name_trial)
        elif base_ckpt_dir:
            current_run_args.checkpoint_dir = os.path.join(base_ckpt_dir, f"trial_{wandb.run.id if wandb.run else 'unknown'}")
        else:
            current_run_args.checkpoint_dir = None
        
        if fabric.global_rank == 0:
            fabric.print(f"--- Starting W&B Sweep Trial (Rank {fabric.global_rank}) ---")
            if wandb.run: fabric.print(f"--- Run Name: {wandb.run.name} (ID: {wandb.run.id}) ---")
            fabric.print("Effective arguments for this trial:")
            for arg_name, value in sorted(vars(current_run_args).items()):
                fabric.print(f"  {arg_name}: {value}")
            if current_run_args.checkpoint_dir:
                os.makedirs(current_run_args.checkpoint_dir, exist_ok=True)
                fabric.print(f"Checkpoints for this trial will be saved to: {current_run_args.checkpoint_dir}")
        
        train_and_evaluate(fabric, current_run_args)

    fabric.launch(train_and_evaluate_sweep_trial_wrapper)
    # wandb.finish() is called automatically by agent or at script end.


if __name__ == "__main__":
    main_parser = get_args_parser()
    cli_args = main_parser.parse_args()

    is_sweep = cli_args.sweep_id is not None or \
               (cli_args.sweep_config_path is not None and os.path.exists(cli_args.sweep_config_path))

    precision_setting = "32-true" if cli_args.mixed_precision == "32" else cli_args.mixed_precision
    
    current_fabric_config = {
        "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
        "devices": "auto",
        "strategy": "ddp" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "auto",
        "precision": precision_setting,
    }

    if is_sweep:
        sweep_run_id = cli_args.sweep_id
        if not sweep_run_id: 
            try:
                with open(cli_args.sweep_config_path, "r", encoding="utf-8") as f:
                    cfg_yaml = yaml.safe_load(f)
                if "project" not in cfg_yaml: cfg_yaml["project"] = cli_args.project_name
                elif cli_args.project_name and cfg_yaml["project"] != cli_args.project_name:
                     print(f"Warn: Project in sweep_config ('{cfg_yaml['project']}') vs --project_name ('{cli_args.project_name}'). Using sweep_config value.")
                
                sweep_run_id = wandb.sweep(cfg_yaml, project=cfg_yaml.get("project", cli_args.project_name))
                print(f"Started new W&B sweep with ID: {sweep_run_id}")
            except Exception as e:
                print(f"Error creating W&B sweep from {cli_args.sweep_config_path}: {e}"); exit(1)
        
        print(f"Attaching W&B agent to sweep ID: {sweep_run_id}")
        wandb.agent(
            sweep_id=sweep_run_id,
            function=lambda: run_sweep_agent_function(base_args=cli_args, fabric_config=current_fabric_config),
            project=cli_args.project_name, 
            count=100, # Max runs for this agent.
        )
    else:  # Single (non-sweep) run
        wandb_logger_instance = None
        if not cli_args.disable_wandb:
            run_name = generate_wandb_run_name(cli_args, is_sweep=False)
            wandb_logger_instance = WandbLogger(
                project=cli_args.project_name, name=run_name, config=vars(cli_args), log_model="all", entity="iamunr4v31",
            )
            # WandbLogger will call wandb.init() when fabric.launch() runs its setup_loggers
        else:
            print("W&B logging is disabled.")
            
        fabric_instance = L.Fabric(
            accelerator=current_fabric_config["accelerator"],
            devices=current_fabric_config["devices"],
            strategy=current_fabric_config["strategy"],
            precision=current_fabric_config["precision"],
            loggers=[wandb_logger_instance] if wandb_logger_instance else [],
        )
        
        # fabric.launch() will execute train_and_evaluate on all processes.
        # The fabric_instance above is passed implicitly to the launched function's context.
        # Or, more explicitly, fabric.launch can take the function and its args.
        def single_run_wrapper(fabric_instance):
            # This code runs inside each process launched by fabric.launch()
            if fabric_instance.global_rank == 0:
                if wandb.run: # wandb.run is initialized by WandbLogger during Fabric setup
                     print(f"--- Starting Single W&B Run: {wandb.run.name} (ID: {wandb.run.id}) ---")
                else:
                     print(f"--- Starting Single Run (W&B Disabled or Error) ---")
                print("Effective arguments for this run:")
                for arg_n, val in sorted(vars(cli_args).items()): print(f"  {arg_n}: {val}")
                if cli_args.checkpoint_dir:
                    os.makedirs(cli_args.checkpoint_dir, exist_ok=True)
                    print(f"Checkpoints will be saved to: {cli_args.checkpoint_dir}")
            train_and_evaluate(fabric_instance, cli_args)

        fabric_instance.launch(single_run_wrapper)

    # Ensure W&B run is finished if it was a single run and W&B was active
    if fabric_instance.global_rank == 0 and not is_sweep and wandb.run is not None: # fabric_instance from single run
        wandb.finish()