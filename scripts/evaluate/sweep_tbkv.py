#!/usr/bin/env python3
"""
Run TBKV configurations (raw + merged at varying merge ratios) and save a
comprehensive statistics JSON for later plotting.

Saved per configuration:
  - top_1, top_5 accuracy
  - total FLOPs per video (caching + matching)
  - avg cache size in MB per video  (K+V tensors, summed across all blocks,
      using the final accumulated cache value at the end of the caching pass)
  - avg memory read from cache per matching frame per video
      (bytes fetched when looking up n_unique_cache_used K/V entries,
       summed across all blocks, averaged over matching frames)
  - avg KV reuse FLOPs per matching frame per video
      (saved K/V projection FLOPs, summed across blocks, averaged over frames)

Usage:
    python scripts/evaluate/sweep_tbkv.py [--n_items N] [--out PATH]
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from omegaconf import OmegaConf

from datasets.kinetics400 import Kinetics400
from models.tbkv_vivit import TBKVFactorizedViViT
from utils.config import load_config
from utils.evaluate_tbkv import evaluate_vivit_metrics

CONFIG_DIR  = Path("configs",  "evaluate", "vivit_kinetics400")
DEFAULT_OUT = Path("results",  "evaluate", "vivit_kinetics400", "sweep_statistics.json")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _set_nested(d, dotted_key, value):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _load_config(name, overrides, n_items):
    cfg_obj = load_config(CONFIG_DIR / f"{name}.yml", to_container=False)
    OmegaConf.update(cfg_obj, "_name", name, merge=True)
    config = OmegaConf.to_container(cfg_obj, resolve=True)
    config["n_items"] = n_items
    for key_path, val in overrides.items():
        _set_nested(config, key_path, val)
    return config


def _build_model(config, device):
    model = TBKVFactorizedViViT(**(config["model"]))
    sd = torch.load(config["weights"], map_location="cpu")
    missing, _ = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "qkv" not in k]
    if real_missing:
        print(f"  [warn] missing keys: {real_missing}")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Stats extraction from _block_stats
# ---------------------------------------------------------------------------

def _extract_run_stats(results):
    """
    Distil the raw evaluation results into the numbers we care about.
    Returns a flat dict.
    """
    block_stats   = results.get("_block_stats",   [])   # list[dict[block_name, list[frame_stat]]]
    flops_summary = results.get("_flops_summary", {})
    metrics       = results.get("metrics",        {})
    n_eval        = results.get("_n_evaluated",   0)

    # -- FLOPs --
    def _sum_counts(c):
        return float(sum(c.values())) if c else 0.0

    total_flops_per_video = _sum_counts(flops_summary.get("total", {}))

    # -- per-video summaries from block_stats --
    # For each video we have {block_name: [frame-stat dicts]}.
    # Cache size: take the *last* caching frame's cache_kv_size_mb for each block
    #   (it's the largest since we accumulate), sum across blocks.
    # Memory read / KV reuse: sum across blocks, average over matching frames.
    cache_mb_per_video           = []
    cache_total_read_bytes_per_vid = []   # summed over ALL matching frames
    cache_mem_read_bytes_per_vid   = []   # per-frame average
    kv_reuse_flops_per_vid         = []

    for vid_stats in block_stats:
        total_cache_mb      = 0.0
        match_read_bytes    = []   # one entry per matching frame (summed over blocks)
        match_kv_reuse      = []   # one entry per matching frame (summed over blocks)

        # Collect per-frame sums across blocks
        # We need to align frames; simpler: collect lists per block and then zip
        block_match_reads  = {}
        block_match_reuses = {}

        for bname, frames in vid_stats.items():
            caching_frames  = [f for f in frames if f['phase'] == 'caching']
            matching_frames = [f for f in frames if f['phase'] == 'matching']

            # Final cache MB for this block
            if caching_frames:
                total_cache_mb += caching_frames[-1].get('cache_kv_size_mb', 0.0)

            # Per matching-frame read bytes and KV-reuse FLOPs for this block
            block_match_reads[bname]  = [f.get('cache_kv_read_bytes',   0) for f in matching_frames]
            block_match_reuses[bname] = [f.get('saved_kv_linear_flops', 0) for f in matching_frames]

        cache_mb_per_video.append(total_cache_mb)

        # Sum across blocks per frame (all blocks should have same # of matching frames)
        n_match_frames = max((len(v) for v in block_match_reads.values()), default=0)
        for fi in range(n_match_frames):
            r = sum(v[fi] for v in block_match_reads.values()  if fi < len(v))
            u = sum(v[fi] for v in block_match_reuses.values() if fi < len(v))
            match_read_bytes.append(r)
            match_kv_reuse.append(u)

        # Total reads across all matching frames for this video
        cache_total_read_bytes_per_vid.append(float(sum(match_read_bytes)))
        cache_mem_read_bytes_per_vid.append(
            sum(match_read_bytes) / len(match_read_bytes) if match_read_bytes else 0.0
        )
        kv_reuse_flops_per_vid.append(
            sum(match_kv_reuse) / len(match_kv_reuse) if match_kv_reuse else 0.0
        )

    def _avg(lst):
        return float(sum(lst) / len(lst)) if lst else 0.0

    avg_cache_mb = _avg(cache_mb_per_video)
    # write bytes = writing full K+V cache once per video
    avg_cache_kv_write_bytes = avg_cache_mb * 1e6
    avg_cache_kv_total_read_bytes = _avg(cache_total_read_bytes_per_vid)

    return {
        "n_evaluated":                    n_eval,
        "top_1":                          float(metrics.get("top_1", 0)),
        "top_5":                          float(metrics.get("top_5", 0)),
        "total_flops_per_video":          total_flops_per_video,
        "flops_caching_per_video":        _sum_counts(flops_summary.get("caching", {})),
        "flops_matching_per_video":       _sum_counts(flops_summary.get("matching", {})),
        "avg_cache_kv_mb":                avg_cache_mb,
        "avg_cache_kv_write_bytes":        avg_cache_kv_write_bytes,
        "avg_cache_kv_total_read_bytes":   avg_cache_kv_total_read_bytes,
        "avg_total_memory_ops_bytes":      avg_cache_kv_write_bytes + avg_cache_kv_total_read_bytes,
        "avg_cache_kv_read_bytes_per_matching_frame":
                                          _avg(cache_mem_read_bytes_per_vid),
        "avg_kv_reuse_flops_per_matching_frame":
                                          _avg(kv_reuse_flops_per_vid),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(n_items, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Videos per run: {n_items}\n")

    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )

    sweep_results = {}

    # ---- tbkv_raw ----
    tag = "tbkv_raw"
    print(f"{'='*55}\nRunning: {tag}\n{'='*55}")
    config = _load_config("tbkv_raw", {}, n_items)
    model  = _build_model(config, device)
    results = evaluate_vivit_metrics(device, model, data, config)
    stats   = _extract_run_stats(results)
    stats["config"] = tag
    stats["merge_ratio"] = None
    sweep_results[tag] = stats
    print(f"  top_1={stats['top_1']:.4f}  FLOPs={stats['total_flops_per_video']:.3e}"
          f"  cache_mb={stats['avg_cache_kv_mb']:.2f}\n")

    # ---- tbkv with merge ratios ----
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        tag = f"tbkv_r{r}"
        print(f"{'='*55}\nRunning: tbkv  merge_ratio={r}\n{'='*55}")
        overrides = {
            "model.spatial_config.block_config.local_merge_ratio":  r,
            "model.temporal_config.block_config.local_merge_ratio": r,
        }
        config  = _load_config("tbkv", overrides, n_items)
        model   = _build_model(config, device)
        results = evaluate_vivit_metrics(device, model, data, config)
        stats   = _extract_run_stats(results)
        stats["config"]      = "tbkv"
        stats["merge_ratio"] = r
        sweep_results[tag] = stats
        print(f"  top_1={stats['top_1']:.4f}  FLOPs={stats['total_flops_per_video']:.3e}"
              f"  cache_mb={stats['avg_cache_kv_mb']:.2f}\n")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Sweep statistics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run_sweep(n_items=args.n_items, out_path=args.out)
