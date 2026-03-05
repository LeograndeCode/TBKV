#!/usr/bin/env python3
"""
Plot Top-1 accuracy (y) vs total FLOPs (x) for TBKV variants:
  - One point for tbkv_raw  (raw cache, no merging)
  - Five points for tbkv    (local_merge_ratio in {0.1, 0.2, 0.3, 0.4, 0.5})

Usage:
    python scripts/evaluate/plot_tbkv_flops_accuracy.py [--n_items N]

Default n_items=100 videos per evaluation point.
"""

import sys
import argparse
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from datasets.kinetics400 import Kinetics400
from models.tbkv_vivit import TBKVFactorizedViViT
from utils.config import load_config
from utils.evaluate_tbkv import evaluate_vivit_metrics

OUTPUT_DIR = Path("results", "evaluate", "vivit_kinetics400")
CONFIG_DIR = Path("configs", "evaluate", "vivit_kinetics400")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_nested(d, dotted_key, value):
    """Set d[a][b][c] = value from 'a.b.c'."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _sum_flops(counts_dict):
    """Sum all FLOPs values in the counts dict returned by evaluate_vivit_metrics."""
    if not counts_dict:
        return 0.0
    return float(sum(counts_dict.values()))


def run_evaluation(config_name, overrides, data, device, n_items):
    """
    Load config, apply overrides, build model, run evaluate_vivit_metrics.
    Returns (total_flops: float, top1: float).
    """
    # Load without resolving so we can inject _name before ${_name} is expanded
    cfg_obj = load_config(CONFIG_DIR / f"{config_name}.yml", to_container=False)
    OmegaConf.update(cfg_obj, "_name", config_name, merge=True)
    config = OmegaConf.to_container(cfg_obj, resolve=True)
    config["n_items"] = n_items

    for key_path, val in overrides.items():
        _set_nested(config, key_path, val)

    model = TBKVFactorizedViViT(**(config["model"]))
    sd = torch.load(config["weights"], map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "qkv" not in k]
    if real_missing:
        print(f"  [warn] missing keys: {real_missing}")

    model = model.to(device).eval()

    results = evaluate_vivit_metrics(device, model, data, config)

    total_flops = _sum_flops(results.get("counts", {}))
    top1 = float(results["metrics"]["top_1"])
    return total_flops, top1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_accuracy_vs_flops(n_items=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Videos per run: {n_items}\n")

    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )

    # --- raw cache (single point) ---
    print("=" * 50)
    print("Running: tbkv_raw")
    print("=" * 50)
    raw_flops, raw_top1 = run_evaluation("tbkv_raw", {}, data, device, n_items)
    print(f"  -> FLOPs={raw_flops:.3e}  top1={raw_top1:.4f}\n")

    # --- tbkv with varying merge ratios ---
    merge_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    tbkv_points = []

    for r in merge_ratios:
        print("=" * 50)
        print(f"Running: tbkv  local_merge_ratio={r}")
        print("=" * 50)
        overrides = {
            "model.spatial_config.block_config.local_merge_ratio":  r,
            "model.temporal_config.block_config.local_merge_ratio": r,
        }
        flops, top1 = run_evaluation("tbkv", overrides, data, device, n_items)
        tbkv_points.append((flops, top1, r))
        print(f"  -> FLOPs={flops:.3e}  top1={top1:.4f}\n")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    tbkv_flops  = [p[0] for p in tbkv_points]
    tbkv_acc    = [p[1] for p in tbkv_points]

    ax.plot(tbkv_flops, tbkv_acc, "o-", color="steelblue",
            linewidth=1.8, markersize=7, label="TBKV (merged cache)", zorder=3)
    for flops, acc, r in tbkv_points:
        ax.annotate(f"r={r}", (flops, acc),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.scatter([raw_flops], [raw_top1], marker="*", s=250,
               color="crimson", zorder=4, label="TBKV (raw cache)")
    ax.annotate("raw", (raw_flops, raw_top1),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Total FLOPs (per video, averaged over matching pass)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy", fontsize=11)
    ax.set_title("Accuracy vs FLOPs — TBKV variants (Kinetics-400)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "flops_accuracy.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    # Also save the raw numbers next to the plot
    csv_path = OUTPUT_DIR / "flops_accuracy.csv"
    with open(csv_path, "w") as f:
        f.write("config,merge_ratio,total_flops,top1\n")
        f.write(f"tbkv_raw,none,{raw_flops},{raw_top1}\n")
        for flops, acc, r in tbkv_points:
            f.write(f"tbkv,{r},{flops},{acc}\n")
    print(f"Numbers saved to {csv_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=100,
                        help="Number of validation videos per evaluation run")
    args = parser.parse_args()
    plot_accuracy_vs_flops(n_items=args.n_items)
