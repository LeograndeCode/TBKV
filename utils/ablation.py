"""
Ablation utilities for sweeping TBKV hyperparameters.

Provides run_evaluations() which loops over a Cartesian product of
merge_values × r_match_values, evaluates the model for each combination,
and writes a summary CSV to the output directory.

Usage (from vivit_kinetics400_ablation.py):
    from utils.ablation import run_evaluations, evaluate_vivit_metrics
"""

import copy
import csv
from pathlib import Path

import torch

from src.utils.misc import get_pytorch_device
from utils.evaluate import evaluate_vivit_metrics  # reuse the two-pass eval loop

# Default sweep values
MERGE_VALUES   = [0.25, 0.5, 0.75, 0.9]
R_MATCH_VALUES = [0.25, 0.5, 0.75, 0.95, 1.00]


def _set_block_param(config: dict, key: str, value) -> dict:
    """Return a deep-copied config with block_config.<key>=value on both branches."""
    cfg = copy.deepcopy(config)
    for branch in ("spatial_config", "temporal_config"):
        try:
            cfg["model"][branch]["block_config"][key] = value
        except KeyError:
            pass
    return cfg


def run_evaluations(config, model_class, data, evaluate_function):
    """
    Sweep merge_value × r_match and write results per combo.

    For each combination a fresh model is instantiated from the same weights
    so there is no state leakage between runs.
    """
    device = config.get("device", get_pytorch_device())
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    merge_values   = config.get("merge_values",   MERGE_VALUES)
    r_match_values = config.get("r_match_values", R_MATCH_VALUES)

    output_dir = Path(config["_output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    total = len(merge_values) * len(r_match_values)
    done  = 0

    for merge in merge_values:
        for r_match in r_match_values:
            done += 1
            print(
                f"\n[ablation {done}/{total}]  "
                f"local_merge_ratio={merge}  r_match={r_match}",
                flush=True,
            )

            # Build a config variant and a fresh model for this combo
            cfg = _set_block_param(config, "local_merge_ratio", merge)
            cfg = _set_block_param(cfg,    "r_match",           r_match)

            model = model_class(**(cfg["model"]))
            sd = torch.load(cfg["weights"], map_location="cpu")
            model.load_state_dict(sd, strict=False)
            model = model.to(device).eval()

            results = evaluate_function(device, model, data, cfg)

            metrics = results.get("metrics", {})
            counts  = results.get("counts",  {})

            row = {
                "local_merge_ratio": merge,
                "r_match":           r_match,
                **{k: round(float(v), 6) for k, v in metrics.items()},
                **{k: round(float(v), 6) for k, v in counts.items()},
            }
            summary_rows.append(row)

            # Print per-combo summary
            top1 = metrics.get("top_1", 0) * 100
            lf   = counts.get("linear_flops", 0)
            print(
                f"  Top-1={top1:.1f}%  linear_flops={lf:.3e}",
                flush=True,
            )

    # Write summary CSV
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        summary_path = output_dir / "ablation_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[ablation] Summary written to {summary_path}")

        # Convenience: sort by top_1 descending
        sorted_rows = sorted(summary_rows, key=lambda r: r.get("top_1", 0), reverse=True)
        with open(output_dir / "ablation_sorted_by_top1.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)

        # Sort by linear_flops ascending
        sorted_flops = sorted(summary_rows, key=lambda r: r.get("linear_flops", float("inf")))
        with open(output_dir / "ablation_sorted_by_linear_flops.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_flops)
