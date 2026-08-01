import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.base import dict_csv_header, dict_csv_line, dict_string
from src.utils.misc import (
    TopKAccuracy,
    get_device_description,
    get_pytorch_device,
    tee_print,
)


# ---------------------------------------------------------------------------
# Block stats collection helpers
# ---------------------------------------------------------------------------

def _collect_and_clear_block_stats(model):
    """Snapshot _frame_stats from PSM-like blocks, then clear them."""
    out = {}
    for name, module in model.named_modules():
        # Avoid strict isinstance checks; class identity can differ if the same
        # file is imported via multiple module paths.
        if hasattr(module, "_frame_stats") and hasattr(module, "cache") and hasattr(module, "caching"):
            out[name] = list(getattr(module, "_frame_stats", []))
            module._frame_stats = []
    return out


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _aggregate_caching_stats(all_video_stats):
    """
    all_video_stats: list (one per video) of {block_name: [caching-frame dicts]}
    Returns {block_name: aggregated_dict}
    """
    if not all_video_stats:
        return {}
    result = {}
    for name in all_video_stats[0]:
        frames = [f for vid in all_video_stats for f in vid.get(name, [])]
        if not frames:
            continue
        result[name] = {
            "avg_fg_tokens":     _mean([f.get("n_fg", 0) for f in frames]),
            "avg_bg_tokens":     _mean([f.get("n_bg", 0) for f in frames]),
            # final_cache_size is stable after all caching frames of the last video
            "final_cache_size":  frames[-1].get("cache_size", 0),
            "avg_merged_tokens": _mean([f.get("merged_tokens", 0) for f in frames]),
            # merge factor: how many original bg tokens each cached token represents
            "avg_merge_factor":  _mean([
                f["n_bg"] / f["merged_tokens"]
                for f in frames if f.get("merged_tokens", 0) > 0
            ]),
            "cache_kv_mb":       frames[-1].get("cache_kv_size_mb", 0.0),
        }
    return result


def _aggregate_matching_stats(all_video_stats):
    """
    all_video_stats: list (one per video) of {block_name: [matching-frame dicts]}
    Returns {block_name: aggregated_dict}
    """
    if not all_video_stats:
        return {}
    result = {}
    for name in all_video_stats[0]:
        frames = [f for vid in all_video_stats for f in vid.get(name, [])]
        if not frames:
            continue
        result[name] = {
            "avg_matched_tokens":     _mean([f.get("n_bg_matched", 0) for f in frames]),
            "avg_cache_entries_used": _mean([f.get("n_unique_cache_used", 0) for f in frames]),
            # fg + unmatched_bg + cache entries retrieved = total tokens seen by attention
            "avg_final_tokens":       _mean([
                f.get("n_fg", 0) + f.get("n_bg_unmatched", 0) + f.get("n_unique_cache_used", 0)
                for f in frames
            ]),
            "total_saved_kv_flops":    sum(f.get("saved_kv_linear_flops", 0) for f in frames),
            "total_match_overhead":    sum(f.get("matching_overhead_flops", 0) for f in frames),
        }
    return result


# ---------------------------------------------------------------------------
# Formatted printing helpers
# ---------------------------------------------------------------------------

def _print_section(title, tee_file):
    sep = "=" * (len(title) + 4)
    tee_print(sep, tee_file)
    tee_print(f"  {title}", tee_file)
    tee_print(sep, tee_file)


def _print_caching_stats(avg_counts, block_stats, n_cache_frames, tee_file):
    _print_section("CACHING PASS", tee_file)
    tee_print(f"  Frames used for caching : {n_cache_frames}", tee_file)
    tee_print("  FLOPs breakdown (caching mechanism, avg per video):", tee_file)
    tee_print(dict_string(avg_counts), tee_file)
    cache_gflops = sum(float(v) for v in avg_counts.values()) / 1e9
    tee_print(f"  Total GFLOPs (caching, for completeness only): {cache_gflops:.2f}", tee_file)
    tee_print("", tee_file)
    tee_print("  Per-block stats  (averaged over caching frames × videos):", tee_file)
    for bname, s in block_stats.items():
        tee_print(f"    [{bname}]", tee_file)
        tee_print(f"      avg foreground tokens          : {s['avg_fg_tokens']:.1f}", tee_file)
        tee_print(f"      avg background tokens          : {s['avg_bg_tokens']:.1f}", tee_file)
        tee_print(f"      cached tokens (total, final)   : {s['final_cache_size']}", tee_file)
        tee_print(f"      avg merged tokens added/frame  : {s['avg_merged_tokens']:.1f}", tee_file)
        tee_print(f"      avg merge factor  (bg→merged)  : {s['avg_merge_factor']:.2f}x", tee_file)
        tee_print(f"      cache K+V memory               : {s['cache_kv_mb']:.3f} MB", tee_file)
    tee_print("", tee_file)


def _print_matching_stats(avg_counts, block_stats, top1, top5,
                          total_saved_kv, total_match_overhead,
                          avg_matching_frames, kv_saved_per_frame,
                          matching_cost_per_frame, net_saved_per_frame,
                          tee_file):
    _print_section("MATCHING PASS", tee_file)
    tee_print(f"  Top-1 Accuracy              : {top1 * 100:.2f}%", tee_file)
    tee_print(f"  Top-5 Accuracy              : {top5 * 100:.2f}%", tee_file)
    tee_print("", tee_file)
    tee_print("  FLOPs breakdown (matching pass, avg per video):", tee_file)
    tee_print(dict_string(avg_counts), tee_file)
    match_gflops = sum(float(v) for v in avg_counts.values()) / 1e9
    tee_print(f"  Total GFLOPs (MATCHING = reported metric)    : {match_gflops:.2f}", tee_file)
    tee_print("", tee_file)
    tee_print(f"  Matching algorithm FLOPs    : {total_match_overhead:.4e}", tee_file)
    tee_print(f"  KV FLOPs saved vs. baseline : {total_saved_kv:.4e}", tee_file)
    tee_print(f"  Net FLOPs savings           : {total_saved_kv - total_match_overhead:.4e}", tee_file)
    tee_print("", tee_file)
    tee_print("  Per-frame projection terms (matching pass):", tee_file)
    tee_print(f"    avg matching frames/video : {avg_matching_frames:.2f}", tee_file)
    tee_print(f"    KV FLOPs saved / frame    : {kv_saved_per_frame:.4e}", tee_file)
    tee_print(f"    matching cost FLOPs/frame : {matching_cost_per_frame:.4e}", tee_file)
    tee_print(f"    net FLOPs savings / frame : {net_saved_per_frame:.4e}", tee_file)
    tee_print("    projection formula         : net_saved(F) = F * net_saved_per_frame", tee_file)
    tee_print("", tee_file)
    tee_print("  Per-block stats  (averaged over matching frames × videos):", tee_file)
    for bname, s in block_stats.items():
        tee_print(f"    [{bname}]", tee_file)
        tee_print(f"      avg tokens matched (bg reused)   : {s['avg_matched_tokens']:.1f}", tee_file)
        tee_print(f"      avg cache entries used            : {s['avg_cache_entries_used']:.1f}", tee_file)
        tee_print(f"      avg final tokens (fg+bg+cached)  : {s['avg_final_tokens']:.1f}", tee_file)
    tee_print("", tee_file)

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_csv_results(results, output_dir, first_run=False):
    for key, val in results.items():
        if not isinstance(val, dict):
            continue
        with open(output_dir / f"{key}.csv", "a") as csv_file:
            if first_run:
                print(dict_csv_header(val), file=csv_file)
            print(dict_csv_line(val), file=csv_file)

