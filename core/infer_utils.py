"""
core.infer_utils
================
Reusable helpers for MSM inference.  All functions are *stateless*;
`cfg` carries every option.

Assumes three merged dicts exist in `cfg`:
  cfg["base"], cfg["noise"], cfg["modulation"]
and a top-level convenience key cfg["mask_strategy"].
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple
from core.engine import _gather_masked_targets
from core.infer_imp import add_awgn_vec, add_middleton_a_vec

def resolve_checkpoint(cfg):
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / cfg["checkpoint"]).expanduser()

def build_noise_fn(noise_cfg):
    """
    Factory that returns a `callable` adding the requested noise model.

    Args:
        noise_cfg : dict
            {
            "type": "awgn" | "middleton_a",
            "snr_db": [0, 5, 10, ...], # list (passed separately)
            # the next two keys exist only for middleton_a
            "A": 1e-2,
            "gamma": 1e-3,
            }

    Returns:
        fn(waveform_t, snr_db) -> waveform_t_noisy
            waveform_t : torch.FloatTensor  [B, 2, T]  (GPU or CPU)
            snr_db: scalar int/float   per call
    """

    ntype = noise_cfg["type"].lower()

    if ntype == "awgn":
        def _awgn_torch_adapter(wav_t, snr_db):
            wav_cpu = wav_t.detach().cpu().numpy()
            wav_cpu = add_awgn_vec(wav_cpu, snr=snr_db)
            wav_t = torch.as_tensor(wav_cpu, device=wav_t.device,
                                dtype=wav_t.dtype)
            
            # create an all-False mask
            mask_t = torch.zeros_like(wav_t[..., 0, :], dtype=torch.bool)
            return wav_t, mask_t
        return _awgn_torch_adapter

    if ntype == "middleton_a":
        A, gamma = noise_cfg["A"], noise_cfg["gamma"]

        def _midA_torch_adapter(wav_t, snr_db):
            *batch_shape, _, N = wav_t.shape

            wav_cpu = wav_t.detach().cpu().numpy()
            wav_cpu, m = add_middleton_a_vec(wav_cpu, 
                        snr=snr_db, A=A, gamma=gamma, return_m=True)
            
            # masking logic
            SPS=8 # TODO: READ THIS FROM CONFIG OR PASS AS ARGUMENT
            imp_sample = m > 0
            imp_sym = imp_sample.reshape(*batch_shape, N//SPS, SPS).any(axis=-1)
            mask_np = np.repeat(imp_sym, SPS, axis=-1)

            wav_t = torch.as_tensor(wav_cpu, device=wav_t.device, dtype=wav_t.dtype)
            mask_t = torch.as_tensor(mask_np, device=wav_t.device, dtype=torch.bool)

            return wav_t, mask_t
        return _midA_torch_adapter

    if ntype == 'none':
        def _identity(wav_t, snr_db=None):
            mask_t = torch.zeros_like(wav_t[..., 0, :], dtype=torch.bool)
            return wav_t, mask_t
        return _identity
        
    raise ValueError(f"Unknown noise type: {noise_cfg['type']}")

def pool_all_symbol_embeddings(x_encoded, spans, extra_emb=None):
    """
    Mean-pool every symbol span. Shape:
        x_encoded: [B, T, D]
        returns: [N_sym, D]
    """
    B, _, D = x_encoded.shape
    if extra_emb is None:
        extra_emb = torch.zeros(B, D, device=x_encoded.device)

    all_emb = []
    all_idx = []
    for b in range(B):
        for j, (s, e) in enumerate(spans[b]):
            emb = x_encoded[b, s:e].mean(0) + extra_emb[b]
            all_emb.append(emb)
            all_idx.append((b, j))
    return torch.stack(all_emb), all_idx

@torch.no_grad()
def infer_step(encoder: torch.nn.Module,
               classifier: torch.nn.Module,
               batch: Tuple,
               device: torch.device,
               mask_strategy: str = "m15") -> Dict[str, float]:
    """
    Args:
        mask_strategy: "m15" | "no_mask" | "custom"
    """
    encoder.eval()
    classifier.eval()

    x, target, mask, *rest = batch
    x, target = x.to(device), target.to(device)

    # Masking
    if mask_strategy == "m15":
        sample_mask = mask["sample_mask"].to(device)
        x_masked = x.masked_fill(sample_mask.unsqueeze(1), 0.0)

    elif mask_strategy == "custom":
        sample_mask = mask["sample_mask"].to(device)  # Boolean (B, N)
        x_masked = x.masked_fill(sample_mask.unsqueeze(1), 0.0)

    elif mask_strategy == "no_mask":
        x_masked = x
    else:
        raise ValueError(f"unknown mask_strategy {mask_strategy}")

    # Encode
    x_encoded = encoder.encode(x_masked)

    # Pool symbol embeddings
    if mask_strategy in {"m15", "custom"}:
        # Pass the full mask dict so extract_symbol_embeddings
        # can access both sample_mask and symbol_spans
        sym_emb, sym_idx = encoder.extract_symbol_embeddings(
            x_encoded, mask, extra_emb=None)
    else:  # "no_mask"
        sym_emb, sym_idx = pool_all_symbol_embeddings(
            x_encoded, mask["symbol_spans"])

    assert sym_emb is not None, "No symbol embeddings produced—check mask settings"

    # Classifier & Metrics
    logits = classifier(sym_emb)
    tgt = _gather_masked_targets(target, sym_idx) \
          if mask_strategy in {"m15", "custom"} else target.view(-1)

    pred   = logits.argmax(dim=1)
    n_err  = (pred != tgt).sum().item()
    n_sym  = tgt.numel()
    return {"ser": n_err / n_sym, "n_err": n_err, "n_sym": n_sym}

# Plot utilities
def plot_ser_curve(res_dict, out_path):
    snr, ser = zip(*[(k, v["ser"]) for k, v in sorted(res_dict.items())])
    plt.figure()
    plt.semilogy(snr, ser, marker="o")
    plt.xlabel("SNR (dB)"); plt.ylabel("SER")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
