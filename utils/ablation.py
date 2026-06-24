"""
Ablation utilities for sweeping TBKV hyperparameters.

Sweeps are read from YAML config (preferred keys):
    - local_merge_ratio: [..]
    - r_match: [..]

Backward-compatible aliases are also supported:
    - merge_values
    - r_match_values

Nested configuration is supported as well:
    - ablation.local_merge_ratio
    - ablation.r_match
"""

import copy
import csv
from pathlib import Path

import torch

from src.utils.misc import get_pytorch_device
from utils.evaluate import evaluate_vivit_metrics  # reuse the two-pass eval loop


def _set_block_param(config: dict, key: str, value) -> dict:
    """Return a deep-copied config with block_config.<key>=value on both branches."""
    cfg = copy.deepcopy(config)
    for branch in ("spatial_config", "temporal_config"):
        try:
            cfg["model"][branch]["block_config"][key] = value
        except KeyError:
            pass
    return cfg


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _find_first_present(config: dict, keys):
    for key in keys:
        cur = config
        found = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                found = False
                break
        if found:
            return cur
    return None


def _resolve_sweep_values(config: dict, name: str, aliases, model_fallback_paths):
    value = _find_first_present(config, [f"ablation.{name}", name, *aliases])
    if value is None:
        value = _find_first_present(config, model_fallback_paths)
    if value is None:
        raise ValueError(
            "Missing ablation sweep values for "
            f"'{name}'. Add one of: '{name}', 'ablation.{name}', or one of {aliases}."
        )
    values = _as_list(value)
    if len(values) == 0:
        raise ValueError(f"Ablation sweep list for '{name}' is empty.")
    return values


def run_evaluations(config, model_class, data, evaluate_function):
    """
    Sweep merge_value × r_match and write results per combo.

    For each combination a fresh model is instantiated from the same weights
    so there is no state leakage between runs.
    """
    device = config.get("device", get_pytorch_device())
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    merge_values = _resolve_sweep_values(
        config,
        name="local_merge_ratio",
        aliases=["merge_values"],
        model_fallback_paths=[
            "model.spatial_config.block_config.local_merge_ratio",
            "model.temporal_config.block_config.local_merge_ratio",
        ],
    )
    r_match_values = _resolve_sweep_values(
        config,
        name="r_match",
        aliases=["r_match_values"],
        model_fallback_paths=[
            "model.spatial_config.block_config.r_match",
            "model.temporal_config.block_config.r_match",
        ],
    )

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
