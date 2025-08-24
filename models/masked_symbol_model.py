from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
from reformer_pytorch import Reformer
import math

from data.data_meta import load_config

class PositionalEncoding(nn.Module):
    """Fixed sine-cosine positional encoding.

    A pre-computed matrix P of dimension 1-by-max_len-by-d_model
    is added to the input sequence so the network can distinguish
    sample positions.

    Attributes:
        pe: torch.Tensor
            Buffer holding the positional-encoding matrix.
        max_len: int
            Maximum sequence length in samples.
        d_model: int
            Embedding dimension.
    """
    def __init__(self):
        super().__init__()
        CONFIG = load_config("config/config.yaml")
        self.max_len = CONFIG["n_samples"] # Maximum sequence length in samples.
        self.d_model = CONFIG["embedding_dim"] # Dimension of the embedding vector.

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10e3) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        # Add batch axis -> (1, T, D) so it broadcasts over B.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encodings to an input sequence.

        Args:
            x: torch.Tensor
                Input tensor of shape `[B, T, d_model]`.
    
        Returns:
            torch.Tensor
                Tensor of the same shape `[B, T, d_model]` with positional
                information added.
        """
        return x + self.pe[:, :x.size(1)]
    
class MaskedSymbolTransformer(nn.Module):
    def __init__(self, in_channels:int=2, depth:int=6):
        """Backbone encoder for the Masked-Symbol Model.

        The module performs three steps:
        1. Input Projection / Channel Mixing: A learnable 1x1 convolution
        maps each time-sample from "embedding vector" size of
        'in_channels' to the 'embedding_dim'. Mathematical operation is:
                    h = W*x + b,    W in R^{embedding_dim x in_channels},
                                    x in R^{in_channels x n_samples}.
        This mixes the input channels but introduces no temporal receptive
        field. Temporal context is left to the Reformer layers to learn.

        2. Temporal indexing: Fixed sinusoidal positional encodings are added so the
        network knows each sample's position.

        3. Contextual encoding: A Reformer stack produces contextual embeddings for
        every sample.

        Args:
            in_channels: Number of input waveform channels.
            depth: Number of Reformer blocks.
        """
        super().__init__()
        CONFIG = load_config("config/config.yaml")
        self.seq_len = CONFIG["n_samples"] # (Maximum) sequence length in samples.
        self.dim = CONFIG["embedding_dim"] # Dimension of the embedding vector.
        self.in_channels = in_channels
        self.depth = depth

        self.input_proj = nn.Conv1d(self.in_channels, self.dim, kernel_size=1)  # [B, C, T] -> [B, D, T]
        self.pos_encoder = PositionalEncoding()

        self.reformer = Reformer(
            dim=self.dim,
            depth=self.depth,
            heads=8,
            lsh_dropout=0.1,
            causal=False,
            bucket_size=64,
            n_hashes=4,
            ff_chunks=200,
            weight_tie=True
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return contextual embeddings for each time-sample.

        Args:
            x: Raw I/Q waveform - channel-first layout,
                shape [B, C, T].
        Returns:
            torch.Tensor
                Encoded sequence, shape [B, T, D]
        """
        x_proj = self.input_proj(x)               # [B, D, T]
        x_proj = x_proj.permute(0, 2, 1)          # [B, T, D]
        x_proj = self.pos_encoder(x_proj)
        x_encoded = self.reformer(x_proj)         # [B, T, D]
        return x_encoded

    def extract_symbol_embeddings(self, x_encoded: torch.Tensor, 
                                batched_mask: Dict[str, object], 
                                extra_emb: Optional[torch.Tensor]=None
        ) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int]]]:
        """Aggregate per-symbol embeddings from the encoded sequence.

        Args:
            x_encoded: Output of the 'encode' method, shape `[B, T, D]`.
            batched_mask: Dictionary produced by `collate_fn`.
            extra_emb: Per-batch auxiliary embedding to add, shape `[B, D]`. 
                If `None` a zero tensor is created, meaning no additive bias
                is applied.

        Returns:
            tuple
                symbol_embeddings: torch.Tensor | None
                    Tensor `[N_sym, D]` or `None` if no symbols matched.
                symbol_indices: list[tuple[int, int]]
                    `(batch_idx, symbol_idx)` for every row in
                    symbol_embeddings.
        """
        B, _, D = x_encoded.shape

        if extra_emb is None:
            extra_emb = torch.zeros(B, D, device=x_encoded.device)
        else:
            extra_emb = extra_emb.to(x_encoded.device)

        symbol_embeddings: List[torch.Tensor] = []
        symbol_indices: List[Tuple[int, int]] = []

        for b in range(B):
            spans = batched_mask["symbol_spans"][b]
            mask_flags = batched_mask["symbol_mask_flags"][b]

            for idx, (start, end) in enumerate(spans):
                if not mask_flags[idx]: # Skip un-masked symbols (flag == False)
                    continue
                # Mean-pool across the samples of this symbol
                emb = x_encoded[b, start:end].mean(dim=0) + extra_emb[b]
                symbol_embeddings.append(emb)
                symbol_indices.append((b, idx))
        
        if not symbol_embeddings: # No masked symbols in batch
            return None, []

        return torch.stack(symbol_embeddings), symbol_indices