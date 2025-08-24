import os
import datetime
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.heads import SingleLayerHead
from models.masked_symbol_model import MaskedSymbolTransformer

from core.data_setup import WaveformIterableDataset 
from core.train_utils import (collate_fn, 
                              worker_init_fn,
                              create_run_dirs)
from core.engine import train_step
from data.data_meta import (load_config, 
                            create_vocab,
                            get_vocab_size,
                            get_symbol_counts)

# "Temporary" OpenMP conflict 
# (libiomp5md.dll already initialized) fix
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# Load config
config_dict = load_config()

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
project_name = "Masked-Symbol-Model"
run_date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
wandb.init(project=project_name, 
           name=f"run-{run_date_time}",
           config=config_dict,
           mode="offline")
config = wandb.config
figure_dir = create_run_dirs("figures", project_name, run_date_time)
checkpoint_dir = create_run_dirs("checkpoints", project_name, run_date_time)

# Model
encoder = MaskedSymbolTransformer().to(device)
vocab = create_vocab(config.mod_tuple)
num_classes = get_vocab_size(vocab)
classifier = SingleLayerHead(in_features=config.embedding_dim,
                             out_features=num_classes).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), 
                             lr=config.learning_rate)
symbol_counts = np.array(get_symbol_counts(vocab, config.mod_tuple), dtype=np.float32)
class_weights = 1.0/symbol_counts 
class_weights = num_classes*(class_weights/np.sum(class_weights))
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Dataloaders
dataset = WaveformIterableDataset()
train_loader = DataLoader(dataset, 
                          batch_size=config.train_batch_size, 
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True,
                          worker_init_fn=worker_init_fn,
                          collate_fn=collate_fn)

# Training Loop
step = 0
best_acc = 0
encoder.train()
classifier.train()

pbar = tqdm(train_loader)
for batch in pbar:
    if step >= config.max_steps:
        wandb.finish()
        break

    result = train_step(encoder, classifier, batch, 
                        optimizer, device, loss_fn=criterion)

    wandb.log({
        "step": step,
        "train/loss": result["loss"], 
        "train/acc": result["acc"]},
        step=step)

    # tqdm update
    pbar.set_description(
        f"step {step} | loss {result['loss']:.4f} | acc {result['acc']:.4f}"
    )

    if result["acc"] > best_acc:
        best_acc = result["acc"]
        torch.save({
            "encoder":    encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "optim":      optimizer.state_dict(),
            "step":       step,
            "best_acc":   best_acc,
        }, f"{checkpoint_dir}/best.pt")
        print(f"Saved new best model (acc={best_acc:.4f}) at step {step}")

    step += 1