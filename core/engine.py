import torch
from typing import Dict, List, Tuple

# ------------------------------------------------------------
# Helper: collect target IDs that align with the masked symbols
# ------------------------------------------------------------
def _gather_masked_targets(targets: torch.Tensor,
                           symbol_indices: List[Tuple[int, int]],
    ) -> torch.Tensor:
    """Pick the label at (batch_idx, symbol_idx) for every masked symbol."""
    rows = targets[tuple(zip(*symbol_indices))] 
    return rows 

def train_step(encoder: torch.nn.Module,
               classifier: torch.nn.Module,
               batch: Tuple,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               loss_fn,
    ) -> Dict[str, float]:
    """
    Forward + backward for one mini-batch.

    Returns
    -------
    dict : {"loss": float, "n_mask": int}
    """
    encoder.train()
    classifier.train()
    optimizer.zero_grad()

    x, target, mask, *rest = batch
    x = x.to(device)
    target = target.to(device)

    # Apply sample-wise masking before encoding .
    # mask['sample_mask'] has shape `[B, T]`.
    # unsqueeze(1) inserts a singleton channel axis -> `[B, C, T]`.
    # Now, mask['sample_mask'] has shape `[B, C, T]`.
    # So, its shape aligns with the shape of x: `[B, C, T]`.
    x_masked = x.masked_fill(mask["sample_mask"]
                         .unsqueeze(1)
                         .to(x.device),
                         0.0)

    # Encode and pool symbol embeddings 
    sym_emb, sym_idx = encoder.extract_symbol_embeddings(
        encoder.encode(x_masked), mask, extra_emb=None
    )
    if sym_emb is None: # no masked symbols
        return {"loss": 0.0, "acc": 0.0, "n_mask": 0}

    logits = classifier(sym_emb)
    tgt_mask = _gather_masked_targets(target, sym_idx)
    loss = loss_fn(logits, tgt_mask)
    acc  = (logits.argmax(dim=1) == tgt_mask).float().mean().item()

    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "acc": acc, "n_mask": int(tgt_mask.numel())}
