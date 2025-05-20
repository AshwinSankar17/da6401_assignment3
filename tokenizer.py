import json
from collections import Counter
from typing import List, Dict, Iterable

class CharTokenizer:
    """
    A character-level tokenizer.
    Manages vocabulary, encoding text to sequences of IDs, and decoding IDs back to text.
    Includes special tokens for padding, start-of-sequence, end-of-sequence, and unknown characters.
    """
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, char_to_idx: Dict[str, int] = None, idx_to_char: List[str] = None):
        """
        Initializes the tokenizer.

        Args:
            char_to_idx (Dict[str, int], optional): Pre-defined mapping from characters to indices.
            idx_to_char (List[str], optional): Pre-defined list of characters (vocabulary).
        """
        if char_to_idx is not None and idx_to_char is not None:
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
            self._vocab_size = len(self.idx_to_char)
        else:
            self.char_to_idx: Dict[str, int] = {}
            self.idx_to_char: List[str] = []
            self._vocab_size = 0
            self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        """Initializes special tokens in the vocabulary."""
        special_tokens = [
            (self.PAD_TOKEN, self.PAD_IDX),
            (self.SOS_TOKEN, self.SOS_IDX),
            (self.EOS_TOKEN, self.EOS_IDX),
            (self.UNK_TOKEN, self.UNK_IDX),
        ]
        # Ensure fixed indices for special tokens
        # Pre-allocate space if necessary, assuming special tokens have the lowest indices
        max_special_idx = max(idx for _, idx in special_tokens)
        if len(self.idx_to_char) <= max_special_idx:
            self.idx_to_char.extend([""] * (max_special_idx + 1 - len(self.idx_to_char)))
        
        for token, idx in special_tokens:
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
        self._vocab_size = len(self.idx_to_char)


    def fit(self, corpus: Iterable[str]):
        """
        Builds the vocabulary from a corpus of texts.

        Args:
            corpus (Iterable[str]): A list or iterable of strings.
        """
        if not self.idx_to_char or not self.char_to_idx: # If not pre-initialized
             self._initialize_special_tokens()

        all_chars = Counter()
        for text in corpus:
            all_chars.update(list(text))

        # Start adding new characters after special tokens
        current_idx = len(self.idx_to_char)
        # Sort characters for reproducibility
        sorted_chars = sorted(all_chars.keys())

        for char in sorted_chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = current_idx
                self.idx_to_char.append(char)
                current_idx += 1
        self._vocab_size = len(self.idx_to_char)

    def encode(self, text: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encodes a text string into a list of character IDs.

        Args:
            text (str): The input string.
            add_sos (bool, optional): Whether to prepend SOS token. Defaults to False.
            add_eos (bool, optional): Whether to append EOS token. Defaults to False.

        Returns:
            List[int]: A list of character IDs.
        """
        if not self.char_to_idx:
            raise RuntimeError("Tokenizer has not been fitted or loaded. Call fit() or load_tokenizer().")

        ids = []
        if add_sos:
            ids.append(self.SOS_IDX)
        
        for char in text:
            ids.append(self.char_to_idx.get(char, self.UNK_IDX))
        
        if add_eos:
            ids.append(self.EOS_IDX)
        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        """
        Decodes a list of character IDs back into a string.

        Args:
            ids (List[int]): A list of character IDs.
            remove_special_tokens (bool, optional): Whether to remove special tokens from the decoded string.
                                                    SOS, EOS, PAD will be removed. UNK usually kept.
                                                    Defaults to True.

        Returns:
            str: The decoded string.
        """
        if not self.idx_to_char:
             raise RuntimeError("Tokenizer has not been fitted or loaded. Call fit() or load_tokenizer().")

        chars = []
        for token_id in ids:
            if token_id >= len(self.idx_to_char) or token_id < 0:
                # This case should ideally not happen if vocab is consistent
                chars.append(self.UNK_TOKEN) 
                continue

            char = self.idx_to_char[token_id]
            if remove_special_tokens:
                if char in [self.SOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN]:
                    continue
            chars.append(char)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary (including special tokens)."""
        return self._vocab_size

    def save_tokenizer(self, filepath: str):
        """Saves the tokenizer's vocabulary to a JSON file."""
        data = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {filepath}")

    @classmethod
    def load_tokenizer(cls, filepath: str) -> 'CharTokenizer':
        """Loads the tokenizer's vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert keys back to int if necessary for idx_to_char if it was saved as dict
        # However, idx_to_char is List[str], json handles this fine.
        # char_to_idx keys are strings, also fine.
        instance = cls(char_to_idx=data['char_to_idx'], idx_to_char=data['idx_to_char'])
        print(f"Tokenizer loaded from {filepath}")
        return instance