import numpy as np
from pathlib import Path
import yaml

from data.modulation import modulate

# --------------------
# Config Utilities
# --------------------
def load_config(path: str="config/config.yaml") -> dict:
    config_path = Path(__file__).resolve().parent.parent / path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --------------------
# Vocabulary Utilities
# --------------------
def create_vocab(mod_tuple: list[tuple]) -> dict:
    """Constructs a vocabulary of distinct modulated symbols mapped to token indices.

    Args:
        mod_tuple: List of (mod_fam, M, label) tuples.

    Returns:
        dict: Mapping from symbol tuples to integer token indices.
    """
    vocab = {}
    token_idx = 0
    for mod_fam, M, _ in mod_tuple:
        sig_in = np.arange(M)
        symbols = modulate(sig_in, mod_fam, M)
        for i in range(symbols.shape[1]):
            tup = tuple(symbols[:, i])
            if tup not in vocab:
                vocab[tup] = token_idx
                token_idx += 1
    return vocab


def get_vocab_size(vocab: dict) -> int:
    """Returns the number of unique tokens in the corpus vocabulary.

    Args:
        vocab: Corpus vocabulary.

    Returns:
        int: Number of unique tokens.
    """
    return len(vocab)


def get_symbol_counts(vocab: dict, mod_tuple: list[tuple]) -> list[int]:
    """Computes how many times each token appears across all modulation schemes.

    Args:
        vocab: Corpus vocabulary.
        mod_tuple: List of modulation tuples (mod_fam, M, label).

    Returns:
        list[int]: List of occurrence counts for each token index.
    """
    counts = [0] * len(vocab)
    for mod_fam, M, _ in mod_tuple:
        sig_in = np.arange(M)
        symbols = modulate(sig_in, mod_fam, M)
        for i in range(symbols.shape[1]):
            key = tuple(symbols[:, i])
            if key in vocab:
                counts[vocab[key]] += 1
    return counts

def tokens_from_symbols(symbols: np.ndarray, vocab: dict) -> list[int]:
    """Converts a matrix of symbols to their corresponding token indices.

    Args:
        symbols: Array of shape (2, T) representing I/Q symbols.
        vocab: Mapping from symbol tuples to token indices.

    Returns:
        list[int]: List of token indices corresponding to each symbol column.
    """
    keys = tuple(map(tuple, symbols.T))
    return [vocab[k] for k in keys]