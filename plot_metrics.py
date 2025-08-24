from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

"""
Plot SER results from multiple experiment run folders.

If any of the discovered run‑folder paths contains the substring 
`none_clean`, plot a horizontal bar chart (one bar per modulation
family) because those runs report SER at a single SNR value ("inf").

Otherwise, fall back to the original scatter/line plot that shows SER
versus SNR for each modulation family.
"""

# Directory discovery helpers

def find_runs_by_pattern(root: str | Path, needle: str) -> list[Path]:
    """Return all immediate sub‑directories of `root` whose names contain
    `needle`.
    """
    root = Path(root)
    return sorted([d for d in root.iterdir() if d.is_dir() and needle in d.name])

# Configuration
ROOT = Path("./experiments/infer")
PATTERN = "custom_middleton_a_A0.020315_g1e-06"
# "custom_middleton_a_A0.020315_g1e-06"
# "custom_middleton_a_A0.020315_g0.001"

run_dirs = find_runs_by_pattern(ROOT, PATTERN)
print("Found", len(run_dirs), "runs:")
for p in run_dirs:
    print(" -", p.name)

if not run_dirs:
    raise SystemExit("No runs matched the given pattern — check ROOT/PATTERN.")

# Helpers to load YAML / JSON files produced by each run

def load_cfg(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_metrics(path: Path) -> dict[str, float]:
    with open(path, "r") as f:
        metrics = json.load(f)
    return metrics["avg_ser"]

# Gather all records into a pandas DataFrame: mod | snr_db | ser
records: list[dict] = []
base_cfg: dict | None = None

for run in run_dirs:
    cfg = load_cfg(run / "cfg.yaml")

    # Modulation label 
    if cfg["family"].upper() == "MIXED":
        mod_key = "MIXED"
    else:
        mod_key = f"{cfg['family'].upper()}-{cfg['M']}"  # e.g. PSK-8

    # Filter: keep only runs that share identical non‑modulation config 
    cfg_no_mod = {k: v for k, v in cfg.items() if k not in ("family", "M")}
    if base_cfg is None:
        base_cfg = cfg_no_mod  # first run establishes the reference
    if cfg_no_mod != base_cfg:
        continue  # skip incompatible runs

    # Flatten metrics (one row per SNR value)
    for snr_str, ser in load_metrics(run / "metrics.json").items():
        # Robust SNR parsing: allow "inf" as well as numeric strings
        if str(snr_str).lower() in ("inf", "infty", "infinite"):
            snr_db = float("inf")
        else:
            try:
                snr_db = int(snr_str)
            except ValueError:
                snr_db = float(snr_str)  # fall-back to float for things like "7.5"

        records.append({"mod": mod_key, "snr_db": snr_db, "ser": ser})

if not records:
    raise SystemExit("No records left after filtering for identical channel config.")


df = pd.DataFrame.from_records(records)
use_barh = any("none_clean" in str(p) for p in run_dirs)
fig, ax = plt.subplots(figsize=(8.5, 5))

def _legend_key(lbl: str) -> tuple[int, int]:
    if lbl.startswith("PSK"):
        return (0, int(lbl.split("-")[1]))
    if lbl.startswith("QAM"):
        return (1, int(lbl.split("-")[1]))
    return (2, 0)

style = {
    "PSK-2":  dict(marker="o", linestyle="-"),
    "PSK-4":  dict(marker="s", linestyle="-"),
    "PSK-8":  dict(marker="^", linestyle="-"),
    "PSK-16": dict(marker="D", linestyle="-"),

    "QAM-4":   dict(marker="o", linestyle="--"),
    "QAM-16":  dict(marker="s", linestyle="--"),
    "QAM-64":  dict(marker="^", linestyle="--"),
    "QAM-256": dict(marker="D", linestyle="--"),

    "MIXED": dict(marker="x", linestyle=":"),
}

default_style = dict(marker="o", linestyle="-")  # fallback

if use_barh:
    snr_sel = df["snr_db"].iloc[0]
    snr_label = "Infinity" if np.isinf(snr_sel) else f"{snr_sel} dB"

    df_bar = df.sort_values("ser", ascending=True)

    ax.barh(df_bar["mod"], df_bar["ser"], color="tab:blue")
    ax.set_xlabel("Symbol Error Rate (SER)", fontsize=14)
    ax.set_ylabel("", fontsize=14)

    ax.set_title(f"SER at SNR = {snr_label} (non-impaired signal)",
                 fontweight="bold")

    # Annotate bars with numeric value
    for y, ser in enumerate(df_bar["ser"]):
        ax.text(ser, y, f"{ser:.3f}", va="center", ha="left", fontsize=8)

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()  # best performers on top
else:
    for mod, grp in df.groupby("mod"):
        ax.plot(grp["snr_db"], grp["ser"], label=mod, **style.get(mod, default_style),
                linewidth=1.75)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel("SNR (dB)", fontsize=16)
    ax.set_ylabel("Symbol Error Rate (SER)", fontsize=16)
    ax.set_title("SER vs. SNR", fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda t: _legend_key(t[0]))
    labels_sorted, handles_sorted = zip(*sorted_pairs)
    ax.legend(handles_sorted, labels_sorted, title="Modulation", title_fontsize=14,
              fontsize=14, 
              bbox_to_anchor=(1, 1), loc="upper left")

plt.tight_layout()
plt.show()

fig_dir = Path(__file__).resolve().parent / "figures" / "Masked-Symbol-Model"
fig_dir.mkdir(parents=True, exist_ok=True)

outfile = fig_dir / f"{PATTERN}.svg"
fig.savefig(outfile, dpi=300, bbox_inches="tight")