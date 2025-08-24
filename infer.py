import yaml, json, datetime, torch
from collections import defaultdict
from pathlib import Path

from core.train_utils import collate_fn, set_seed
from data.data_meta import load_config
from core.infer_utils import (resolve_checkpoint, 
                            infer_step,
                            build_noise_fn)
from models.masked_symbol_model import MaskedSymbolTransformer
from models.heads import SingleLayerHead
from core.data_setup import WaveformIterableDataset
from torch.utils.data import DataLoader

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cfg_train = load_config()

# Load config
CFG_DIR = Path(__file__).resolve().parents[0] / "config" / "infer"
cfg_base = yaml.safe_load(open(CFG_DIR/"base.yaml"))
cfg_noise = yaml.safe_load(open(CFG_DIR/"noise.yaml"))
cfg_mod = yaml.safe_load(open(CFG_DIR/"modulation.yaml"))
cfg = {**cfg_base, **cfg_noise, **cfg_mod}

# Create experiment tag
tag = f"{cfg['mask_strategy']}_{cfg['type']}"
if cfg["type"] == "middleton_a":
    tag += f"_A{cfg['A']}_g{cfg['gamma']}"
tag += f"_{cfg['family']}{cfg['M']}"

# Create run directory
run_dir = Path("experiments/infer") / (
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + tag
    )
run_dir.mkdir(parents=True, exist_ok=True)

# Write the merged config
yaml.safe_dump(cfg, open(run_dir/"cfg.yaml", "w"))

# Load the model
ckpt_path = resolve_checkpoint(cfg)
ckpt = torch.load(ckpt_path, map_location="cpu")

encoder = MaskedSymbolTransformer().to(device)
encoder.eval()

out_features = ckpt["classifier"]["linear.weight"].shape[0]
classifier = SingleLayerHead(cfg_train["embedding_dim"], 
                            out_features=out_features).to(device)
classifier.eval()

encoder.load_state_dict(ckpt["encoder"])
classifier.load_state_dict(ckpt["classifier"])

noise_fn = build_noise_fn(cfg_noise)

# Aggregates across seeds
agg_err = defaultdict(int) # total symbol errors for each SNR
agg_sym = defaultdict(int) # total number of symbols for each SNR
seed_results = {} # Stores per-seed results

# Main evaluation loop
for seed in cfg["seeds"]:
    print(f"\nSeed: {seed}")
    set_seed(seed)

    dataset = WaveformIterableDataset(
        mask_rate=0.15 if cfg["mask_strategy"] == "m15" else 0.0,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        collate_fn=collate_fn,
        num_workers=0
    )

    res_seed = {}
    for snr in cfg["snr_db"]:
        n_err = n_sym = 0
        for batch in data_loader:
            x, target, mask, *rest = batch

            wav_noisy, sample_mask_custom = noise_fn(x, snr)
            sample_mask_custom.to(device=device)
            SPS = 8 # TODO: Read this from config.
            noise_sym_mask = (sample_mask_custom
                    .view(*sample_mask_custom.shape[:-1], -1, SPS).any(dim=-1))
            noise_sym_mask.to(device=device)
            mask_dict = {
                "sample_mask": sample_mask_custom,
                "symbol_spans": mask["symbol_spans"],
                "symbol_mask_flags": noise_sym_mask 
            }

            stats = infer_step(
                encoder,
                classifier,
                batch=(wav_noisy, target, mask_dict, *rest),
                device=device,
                mask_strategy=cfg["mask_strategy"]
            )

            n_err += stats["n_err"]
            n_sym += stats["n_sym"]
            if n_sym >= cfg["target_symbols"]:
                break
        ser = n_err / n_sym
        res_seed[snr] = {"ser": ser,
                        "n_err": n_err,
                        "n_sym": n_sym}
        agg_err[snr] += n_err
        agg_sym[snr] += n_sym
    seed_results[f"seed_{seed}"] = res_seed

avg_ser = {snr: (agg_err[snr] / agg_sym[snr]) if agg_sym[snr] 
        else float("nan") for snr in cfg["snr_db"]}

print("\nAveraged SER:")
for snr, ser in avg_ser.items():
    print(f"  SNR={snr:>3} dB -> SER={ser:.4f}")

# save JSON
json.dump({"per_seed": seed_results, 
        "avg_ser": avg_ser}, 
        open(run_dir / "metrics.json", "w"), indent=2)

print("\nFinished", run_dir)