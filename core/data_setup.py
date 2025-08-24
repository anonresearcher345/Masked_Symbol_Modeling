"""Uses modules from 'data/' to prepare the dataset.
"""
import sys
import random
from typing import Dict, Iterator
from pathlib import Path
sys.path.append(str(Path.cwd().parent.joinpath('data')))

from torch.utils.data import IterableDataset
import torch

from data.data_meta import load_config

from data.data_generator import generate_waveform, WaveformExample

class WaveformIterableDataset(IterableDataset):
    def __init__(self, mask_rate: float=0.15, return_attributes: bool=False,
                sps:int | None=None, n_samples: int | None=None):
        """Dataset to yield waveform samples one at a time using generate_waveform.

        Args:
            mask_rate: Fraction of symbols to mask in each example.
            return_attributes: Whether to include clean waveform,
                modulation label, the original symbols, etc.
            sps: Samples per symbol.
            n_samples: Total number of samples per generated example.
        """
        super().__init__()
        self.mask_rate = mask_rate
        self.return_attributes = return_attributes

        CONFIG = load_config("config/config.yaml")
        self.sps =  sps if sps is not None else CONFIG["sps"]
        self.n_samples = n_samples if n_samples is not None else CONFIG["n_samples"]

    def create_sample_mask(self) -> Dict[str, object]:
        """Generate symbol- and sample-level masks for a single waveform. 

        A subset of symbols is chosen at the rate defined by
        'mask_rate'. All time-samples belonging to those symbols are
        set to 'True' in the returned boolean mask.

        The returned dictionary is passed unchanged through the
        DataLoader, allowing the training loop to zero-out the chosen samples
        and to compute losses only on the masked symbols.

        Returns:
            dict:
                sample_mask: torch.BoolTensor
                    'True' at each time-sample to be zeroed before (Transformer) encoding.
                symbol_spans: list[tuple[int, int]]
                    '(start, end)' indices for each symbol in the waveform.
                symbol_mask_flags: list[bool]
                    Boolean flags aligned with 'symbol_spans' indicating
                    which symbols were chosen for masking.
        """
        n_symbols = self.n_samples // self.sps
        num_to_mask = max(0, int(self.mask_rate * n_symbols))
        masked_idx = set(random.sample(range(n_symbols), num_to_mask))
        symbol_mask_flags = [i in masked_idx for i in range(n_symbols)]

        symbol_spans = [(i*self.sps, (i+1)*self.sps) for i in range(n_symbols)]
        sample_mask = torch.zeros(self.n_samples, dtype=torch.bool)
        for flag, (start, end) in zip(symbol_mask_flags, symbol_spans):
            if flag:
                sample_mask[start:end] = True
        mask = dict()
        mask["sample_mask"] = sample_mask
        mask["symbol_spans"] = symbol_spans
        mask["symbol_mask_flags"] = symbol_mask_flags
        return mask

    def __iter__(self) -> Iterator:
        while True:
            example: WaveformExample = generate_waveform(return_attributes=self.return_attributes)

            x = torch.from_numpy(example.impaired).float()  # shape: (2, N)
            target = torch.tensor(example.tokens, dtype=torch.int64) # shape: (1, N/SPS)
            mask = self.create_sample_mask()

            if self.return_attributes:
                metadata = {
                    "clean": torch.from_numpy(example.clean).float(),  # shape: (2, N)
                    "label": torch.tensor(example.label, dtype=torch.int32),
                    "symbols": torch.from_numpy(example.symbols).float() # shape: (2, N/SPS)
                }
                yield x, target, mask, metadata
            else:
                yield x, target, mask