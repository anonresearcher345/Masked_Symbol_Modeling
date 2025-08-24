import numpy as np
from dataclasses import dataclass
from typing import Optional

from data.data_utils import unit_sig_pow

from data.data_meta import (
    load_config,
    create_vocab,
    tokens_from_symbols,
)

from data.modulation import modulate
from data.filters import design_rcos_filt, apply_filt

# --------------------
# SECTION 1: Load Configurable Parameters from YAML
# --------------------

CONFIG = load_config("config/config.yaml")

MOD_TUPLE = [tuple(entry) for entry in CONFIG["mod_tuple"]]
SPS = CONFIG["sps"]
FILT_SPAN_IN_SYM_VECT = CONFIG["filt_span_in_sym"]
BETA_VECT = CONFIG["beta"]
N_SAMPLES = CONFIG["n_samples"]

# --------------------
# SECTION 2: Initialize Vocab and RNGs
# --------------------

corpus_vocab = create_vocab(MOD_TUPLE)

filt_span_rng = np.random.default_rng()
beta_rng = np.random.default_rng()
mod_rng = np.random.default_rng()

# --------------------
# SECTION 4: Output Data Structure
# --------------------

@dataclass
class WaveformExample:
    impaired: np.ndarray                  # Impaired waveform (2, N)
    tokens: list[int]                     # Token indices for each symbol (1, N/SPS)
    clean: Optional[np.ndarray] = None    # Clean baseband waveform (2, N)
    label: Optional[int] = None           # Modulation type label
    symbols: Optional[np.ndarray] = None  # Symbols that form the waveform (2, N/SPS)

# --------------------
# SECTION 5: Waveform Generation
# --------------------

def generate_waveform(return_attributes: bool=False) -> WaveformExample:
    """Generates a waveform and its impaired version, with optional metadata.

    Args:
        return_attributes: If True, also returns the clean waveform, the modulation type label,
        and the symbols.

    Returns:
        WaveformExample containing impaired waveform 
        and tokens corresponding to each distinct symbol constellation, 
        and optionally metadata attributes.
    """
    # Sample parameters
    filt_span = filt_span_rng.choice(FILT_SPAN_IN_SYM_VECT)
    beta = beta_rng.choice(BETA_VECT)
    
    # ------------------------------------------------------------------
    # OVERRIDE
    #
    # For certain inference runs we need every generated waveform to use
    # one fixed modulation scheme (e.g., only QPSK or only 16-QAM).
    # The original design randomly samples the modulation family and
    # order from `config/config.yaml`, and the YAML layer currently
    # offers no simple switch to lock those choices.
    #
    # Until the configuration system is refactored, we hard-code the
    # desired tuple `(family, M, label)` below.  If you modify these
    # values, also update `modulation.yaml` so that the printed tags and
    # downstream bookkeeping stay consistent.
    #
    # TODO: Replace this override with a proper CLI / YAML option once
    #       the data-generation pipeline is fully modular.
    # ------------------------------------------------------------------
    # mod_fam, M, mod_label = "qam", 256, 7

    # ===============================================
    # UNCOMMENT WHILE TRAINING
    # OR DURING INFERENCE USING ALL THE MODULATION
    # TYPES WITH RANDOM SAMPLING
    mod_fam, M, mod_label = mod_rng.choice(MOD_TUPLE)
    # ===============================================

    M = int(M)
    mod_label = int(mod_label)

    # Design the pulse shaping filter. 
    _, h = design_rcos_filt(beta, filt_span, SPS)

    # Generate and modulate symbols
    dec_in = np.random.randint(low=0, high=M, size=int(N_SAMPLES / SPS))
    symbols = modulate(dec_in, mod_fam, M)

    # Apply the pulse shaping filter
    waveform = apply_filt(symbols, filt_span, SPS, h)
    waveform = unit_sig_pow(waveform)

    # May apply impairments
    impaired = waveform

    tokens = tokens_from_symbols(symbols, corpus_vocab)

    if return_attributes:
        return WaveformExample(
            impaired=impaired,
            tokens=tokens,
            clean=waveform,
            label=mod_label,
            symbols=symbols
        )
    else:
        return WaveformExample(
            impaired=impaired,
            tokens=tokens
        )