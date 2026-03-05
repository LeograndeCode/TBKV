#!/usr/bin/env python3
"""
Sweep over the number of caching frames (P) and collect statistics.

For each value of P, the first P frames are used for the caching pass and
the remaining (32 - P) frames are used for matching.  Both the merged-cache
(tbkv) and raw-cache (tbkv_raw) models are evaluated at every P value.

Saved JSON format mirrors sweep_tbkv.py so the same helper functions work.

Usage:
    python scripts/evaluate/sweep_p_frames.py [--n_items N] [--out PATH]
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
DEFAULT_OUT = Path("results",  "evaluate", "vivit_kinetics400", "sweep_p_frames.json")

P_VALUES     = [4, 8, 16, 20, 24, 28]
TOTAL_FRAMES = 32
MIN_MATCH_FRAMES = 4   # require at least this many raw matching frames


# ---------------------------------------------------------------------------
# Shared helpers (same as sweep_tbkv.py)
# ---------------------------------------------------------------------------

def _set_nested(d, dotted_key, value):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _load_config(name, overrides, n_items, n_cache_frames):
    cfg_obj = load_config(CONFIG_DIR / f"{name}.yml", to_container=False)
    OmegaConf.update(cfg_obj, "_name", name, merge=True)
    config = OmegaConf.to_container(cfg_obj, resolve=True)
    config["n_items"] = n_items
    config["n_cache_frames"] = n_cache_frames
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


def _extract_run_stats(results, n_cache_frames):
    block_stats   = results.get("_block_stats",   [])
    flops_summary = results.get("_flops_summary", {})
    metrics       = results.get("metrics",        {})
    n_eval        = results.get("_n_evaluated",   0)

    def _sum_counts(c):
        return float(sum(c.values())) if c else 0.0

    total_flops_per_video = _sum_counts(flops_summary.get("total", {}))

    cache_mb_per_video             = []
    cache_total_read_bytes_per_vid = []
    cache_mem_read_bytes_per_vid   = []
    kv_reuse_flops_per_vid         = []

    for vid_stats in block_stats:
        total_cache_mb   = 0.0
        match_read_bytes = []
        match_kv_reuse   = []
        block_match_reads  = {}
        block_match_reuses = {}

        for bname, frames in vid_stats.items():
            caching_frames  = [f for f in frames if f['phase'] == 'caching']
            matching_frames = [f for f in frames if f['phase'] == 'matching']

            if caching_frames:
                total_cache_mb += caching_frames[-1].get('cache_kv_size_mb', 0.0)

            block_match_reads[bname]  = [f.get('cache_kv_read_bytes',   0) for f in matching_frames]
            block_match_reuses[bname] = [f.get('saved_kv_linear_flops', 0) for f in matching_frames]

        cache_mb_per_video.append(total_cache_mb)

        n_match_frames = max((len(v) for v in block_match_reads.values()), default=0)
        for fi in range(n_match_frames):
            r = sum(v[fi] for v in block_match_reads.values()  if fi < len(v))
            u = sum(v[fi] for v in block_match_reuses.values() if fi < len(v))
            match_read_bytes.append(r)
            match_kv_reuse.append(u)

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
    avg_cache_kv_write_bytes      = avg_cache_mb * 1e6
    avg_cache_kv_total_read_bytes = _avg(cache_total_read_bytes_per_vid)

    return {
        "n_evaluated":             n_eval,
        "n_cache_frames":          n_cache_frames,
        "n_match_frames":          TOTAL_FRAMES - n_cache_frames,
        "top_1":                   float(metrics.get("top_1", 0)),
        "top_5":                   float(metrics.get("top_5", 0)),
        "total_flops_per_video":   total_flops_per_video,
        "flops_caching_per_video": _sum_counts(flops_summary.get("caching", {})),
        "flops_matching_per_video":_sum_counts(flops_summary.get("matching", {})),
        "avg_cache_kv_mb":                        avg_cache_mb,
        "avg_cache_kv_write_bytes":               avg_cache_kv_write_bytes,
        "avg_cache_kv_total_read_bytes":           avg_cache_kv_total_read_bytes,
        "avg_total_memory_ops_bytes":              avg_cache_kv_write_bytes + avg_cache_kv_total_read_bytes,
        "avg_cache_kv_read_bytes_per_matching_frame":  _avg(cache_mem_read_bytes_per_vid),
        "avg_kv_reuse_flops_per_matching_frame":       _avg(kv_reuse_flops_per_vid),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(n_items, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Videos per run: {n_items}\n")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load any previously saved results so we can resume
    if out_path.exists():
        with open(out_path) as f:
            sweep_results = json.load(f)
        print(f"Resuming from {out_path}  ({len(sweep_results)} runs already done)\n")
    else:
        sweep_results = {}

    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )

    for P in P_VALUES:
        M = TOTAL_FRAMES - P
        if M < MIN_MATCH_FRAMES:
            print(f"Skipping P={P}: only {M} matching frames remaining (min={MIN_MATCH_FRAMES})")
            continue
        for cfg_name in ("tbkv", "tbkv_raw"):
            tag = f"{cfg_name}_P{P}"

            if tag in sweep_results:
                print(f"Skipping {tag}  (already in results)")
                continue

            print(f"{'='*55}\nRunning: {cfg_name}  P={P} (cache={P}, match={M})\n{'='*55}")

            config = _load_config(cfg_name, {}, n_items, n_cache_frames=P)
            model  = _build_model(config, device)
            results = evaluate_vivit_metrics(device, model, data, config)
            stats   = _extract_run_stats(results, n_cache_frames=P)
            stats["config"] = cfg_name

            if cfg_name == "tbkv":
                stats["merge_ratio"] = (
                    config["model"]["spatial_config"]["block_config"].get("local_merge_ratio")
                )
            else:
                stats["merge_ratio"] = None

            sweep_results[tag] = stats

            # Save after every run so progress is never lost
            with open(out_path, "w") as f:
                json.dump(sweep_results, f, indent=2)

            print(f"  top_1={stats['top_1']:.4f}  "
                  f"FLOPs={stats['total_flops_per_video']:.3e}  "
                  f"cache_mb={stats['avg_cache_kv_mb']:.2f}\n")

    print(f"Sweep statistics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--out",     type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run_sweep(n_items=args.n_items, out_path=args.out)
