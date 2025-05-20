import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)

from tokenizer import (
    CharTokenizer,
)  # Assuming tokenizer.py is in the same directory or PYTHONPATH


class TransliterationDataset(Dataset):
    """
    Dataset class for transliteration tasks.
    Takes source and target sentences, pre-tokenizes them during initialization,
    and prepares them for the model.
    """

    def __init__(
        self,
        source_sents: List[str],
        target_sents: List[str],
        source_tokenizer: "CharTokenizer",
        target_tokenizer: "CharTokenizer",
        verbose: bool = True,
    ):
        """
        Args:
            source_sents (List[str]): List of source language sentences.
            target_sents (List[str]): List of target language sentences.
            source_tokenizer (CharTokenizer): Tokenizer for the source language.
            target_tokenizer (CharTokenizer): Tokenizer for the target language.
            verbose (bool): Whether to print tokenization progress and use Rich progress bar.
        """
        assert len(source_sents) == len(
            target_sents
        ), "Source and target sentences must have the same length."

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.tokenized_source_sents: List[torch.Tensor] = []
        self.tokenized_target_sents: List[torch.Tensor] = []

        # --- Pre-tokenize all sentences ---
        if verbose:
            progress_columns = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            ]
            with Progress(*progress_columns) as progress:
                src_task = progress.add_task(
                    f"Tokenizing {len(source_sents)} source sentences...",
                    total=len(source_sents),
                )
                for text in source_sents:
                    source_ids = self.source_tokenizer.encode(
                        text, add_sos=True, add_eos=True
                    )
                    self.tokenized_source_sents.append(source_ids)
                    progress.update(src_task, advance=1)

                trg_task = progress.add_task(
                    f"Tokenizing {len(target_sents)} target sentences...",
                    total=len(target_sents),
                )
                for text in target_sents:
                    target_ids = self.target_tokenizer.encode(
                        text, add_sos=True, add_eos=True
                    )
                    self.tokenized_target_sents.append(target_ids)
                    progress.update(trg_task, advance=1)
            print("Pre-tokenization complete.")
        else:  # Fallback to simple print statements if Rich is not used/available
            for i, text in enumerate(source_sents):
                source_ids = self.source_tokenizer.encode(
                    text, add_sos=True, add_eos=True
                )
                self.tokenized_source_sents.append(source_ids)

            if verbose:
                print(f"Pre-tokenizing {len(target_sents)} target sentences...")
            for i, text in enumerate(target_sents):
                target_ids = self.target_tokenizer.encode(
                    text, add_sos=True, add_eos=True
                )
                self.tokenized_target_sents.append(target_ids)

    def __len__(self) -> int:
        return len(self.tokenized_source_sents)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.tokenized_source_sents[idx]), torch.tensor(self.tokenized_target_sents[idx])


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    source_pad_idx: int,
    target_pad_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences in a batch.
    The tensors will be sequence-first (L, N) as commonly used by PyTorch RNNs.
    """
    source_seqs, target_seqs = zip(*batch)

    source_lengths = torch.tensor([s.size(0) for s in source_seqs])
    target_lengths = torch.tensor([t.size(0) for t in target_seqs])

    padded_source_batch = pad_sequence(
        source_seqs, batch_first=False, padding_value=source_pad_idx
    )
    padded_target_batch = pad_sequence(
        target_seqs, batch_first=False, padding_value=target_pad_idx
    )

    return padded_source_batch, source_lengths, padded_target_batch, target_lengths


# class TransliterationDataset(Dataset):
#     """
#     Dataset class for transliteration tasks.
#     Takes source and target sentences, tokenizes them, and prepares them for the model.
#     """

#     def __init__(
#         self,
#         source_sents: List[str],
#         target_sents: List[str],
#         source_tokenizer: CharTokenizer,
#         target_tokenizer: CharTokenizer,
#         verbose: bool = True,
#     ):
#         """
#         Args:
#             source_sents (List[str]): List of source language sentences (e.g., Latin).
#             target_sents (List[str]): List of target language sentences (e.g., Devanagari).
#             source_tokenizer (CharTokenizer): Tokenizer for the source language.
#             target_tokenizer (CharTokenizer): Tokenizer for the target language.
#         """
#         assert len(source_sents) == len(
#             target_sents
#         ), "Source and target sentences must have the same length."
#         self.source_sents = source_sents
#         self.target_sents = target_sents
#         self.source_tokenizer = source_tokenizer
#         self.target_tokenizer = target_tokenizer

#     def __len__(self) -> int:
#         return len(self.source_sents)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Retrieves a single tokenized source-target pair.
#         SOS and EOS tokens are added here.

#         Args:
#             idx (int): Index of the sample.

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]: source_ids_tensor, target_ids_tensor
#         """
#         source_text = self.source_sents[idx]
#         target_text = self.target_sents[idx]

#         # Add SOS and EOS tokens during encoding
#         source_ids = self.source_tokenizer.encode(
#             source_text, add_sos=True, add_eos=True
#         )
#         target_ids = self.target_tokenizer.encode(
#             target_text, add_sos=True, add_eos=True
#         )

#         return torch.tensor(source_ids, dtype=torch.long), torch.tensor(
#             target_ids, dtype=torch.long
#         )


# def collate_fn(
#     batch: List[Tuple[torch.Tensor, torch.Tensor]],
#     source_pad_idx: int,
#     target_pad_idx: int,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Collate function to pad sequences in a batch.
#     The tensors will be sequence-first (L, N) as commonly used by PyTorch RNNs.

#     Args:
#         batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of (source_ids, target_ids) tuples.
#         source_pad_idx (int): Padding index for source sequences.
#         target_pad_idx (int): Padding index for target sequences.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#             - padded_source_batch (L_src, N): Padded source sequences.
#             - source_lengths (N): Original lengths of source sequences.
#             - padded_target_batch (L_trg, N): Padded target sequences.
#             - target_lengths (N): Original lengths of target sequences.
#     """
#     source_seqs, target_seqs = zip(*batch)

#     # Get lengths before padding
#     source_lengths = torch.tensor([len(s) for s in source_seqs], dtype=torch.long)
#     target_lengths = torch.tensor([len(t) for t in target_seqs], dtype=torch.long)

#     # Pad sequences
#     # pad_sequence expects a list of Tensors and pads them to the max length in the list.
#     # It also expects batch_first=False by default, so output is (L, N, *)
#     padded_source_batch = pad_sequence(
#         source_seqs, batch_first=False, padding_value=source_pad_idx
#     )
#     padded_target_batch = pad_sequence(
#         target_seqs, batch_first=False, padding_value=target_pad_idx
#     )

#     return padded_source_batch, source_lengths, padded_target_batch, target_lengths
