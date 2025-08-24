import os
import torch
import numpy as np
import random

def collate_fn(batch):
    """Custom collate function for WaveformIterableDataset.
    Args:
        batch: A list of tuples (x, target, mask, metadata). 
    Returns:
        xs: Tensor (B, 2, N).
        targets: Tensor (B, 1, N/SPS).
        batched_mask: Same keys as the per-exampe 'mask' dict,
            with tensors stacked to shape (B, ...) and
            list-type fields grouped into Python lists.
        batched_metadata: Dict of stacked metadata.
            Collate logic is the same with 'batched_mask'.
    """
    has_meta = len(batch[0]) == 4
    if has_meta:
        xs, targets, masks, metas = zip(*batch)
    else:
        xs, targets, masks = zip(*batch)
        metas = None

    # Stack primary tensors
    xs = torch.stack(xs) # (B, 2, N)
    targets = torch.stack(targets) # (B, 1, N/SPS)

    # Collate mask dict:
    #   Tensor values -> torch.stack([...]) 
    #   Non-tensor values -> list[...]
    batched_mask = dict()
    for key, v in masks[0].items():
        if torch.is_tensor(v):
            batched_mask[key] = torch.stack([m[key] for m in masks])
        else:
            batched_mask[key] = [m[key] for m in masks]

    # Collate optional metadata (same logic)
    if has_meta:
        batched_meta = dict()
        for key, v in metas[0].items():
            if v is None:
                continue
            if torch.is_tensor(v):
                batched_meta[key] = torch.stack([m[key] for m in metas])
            elif isinstance(v, (int, float)):
                batched_meta[key] = torch.tensor([m[key] for m in metas])
            else:
                batched_meta[key] = [m[key] for m in metas]
        
        return xs, targets, batched_mask, batched_meta

    return xs, targets, batched_mask

def worker_init_fn(worker_id: int):
    """Re-seed RNGs for each `DataLoader` worker.

    Ensures that every worker generates independent random numbers
    (e.g., unique `sample_mask`s) while runs remain reproducible when the
    main process is seeded.

    Args:
        worker_id: Identifier assigned by PyTorch (unused here, but
            required by the API).

    Notes:
        PyTorch pre-assigns a 64-bit seed to each worker.  
        This function applies it to Python’s
        `random`, NumPy, and Torch RNGs used inside the dataset.
    """
    import random, numpy as np, torch

    base_seed = torch.initial_seed()
    random.seed(base_seed % 2**32) # Convert to 32-bit
    np.random.seed(base_seed % 2**32)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    # Keep each worker single-threaded to avoid
    # silent core oversubscription.
    # 1 worker process = 1 CPU thread.
    # Comment when num_workers=0.
    # Uncomment otherwise.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def create_run_dirs(base_dir: str, project_name: str, run_date_time: str):
    """Create a run-specific subdirectory under a base directory 
    (e.g., checkpoints or figures).

    Args:
        base_dir: 'figures' or 'checkpoints'
        project_name: wandb project name
        run_time: e.g., '2024-06-09_15-22'

    Returns:
        str: Path to the created directory
    """
    run_dir = os.path.join(base_dir, project_name, run_date_time)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def set_seed(seed: int):
    random.seed(seed) # Set Python's random module seed
    np.random.seed(seed) # Set Numpy seed
    torch.manual_seed(seed) # Set Pytorch CPU  
    if torch.cuda.is_available(): # and (if available) CUDA seeds
        torch.cuda.manual_seed_all(seed)
    
    # Ensure Python's hash-based randomness is fixed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Make PyTorch's cuDNN backend deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False