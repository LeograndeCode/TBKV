import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.base import dict_csv_header, dict_csv_line, dict_string
from src.tbkv.tbkv_blocks import TBKVBlock
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
    """Snapshot _frame_stats from every TBKVBlock, then clear them."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, TBKVBlock):
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
                          total_saved_kv, total_match_overhead, tee_file):
    _print_section("MATCHING PASS", tee_file)
    tee_print(f"  Top-1 Accuracy              : {top1 * 100:.2f}%", tee_file)
    tee_print(f"  Top-5 Accuracy              : {top5 * 100:.2f}%", tee_file)
    tee_print("", tee_file)
    tee_print("  FLOPs breakdown (matching pass, avg per video):", tee_file)
    tee_print(dict_string(avg_counts), tee_file)
    tee_print("", tee_file)
    tee_print(f"  Matching algorithm FLOPs    : {total_match_overhead:.4e}", tee_file)
    tee_print(f"  KV FLOPs saved vs. baseline : {total_saved_kv:.4e}", tee_file)
    tee_print(f"  Net FLOPs savings           : {total_saved_kv - total_match_overhead:.4e}", tee_file)
    tee_print("", tee_file)
    tee_print("  Per-block stats  (averaged over matching frames × videos):", tee_file)
    for bname, s in block_stats.items():
        tee_print(f"    [{bname}]", tee_file)
        tee_print(f"      avg tokens matched (bg reused)   : {s['avg_matched_tokens']:.1f}", tee_file)
        tee_print(f"      avg cache entries used            : {s['avg_cache_entries_used']:.1f}", tee_file)
        tee_print(f"      avg final tokens (fg+bg+cached)  : {s['avg_final_tokens']:.1f}", tee_file)
    tee_print("", tee_file)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_vivit_metrics(device, model, data, config):
    top1 = TopKAccuracy(k=1)
    top5 = TopKAccuracy(k=5)

    data_loader = DataLoader(data, batch_size=1, num_workers=config.get("num_workers", 2))
    n_items        = config.get("n_items", len(data_loader))
    n_cache_frames = config.get("frame_split", 16)

    all_cache_counts = []
    all_match_counts = []
    all_cache_block_stats = []
    all_match_block_stats = []
    n_evaluated = 0

    for idx, (video, label) in tqdm(enumerate(data_loader), total=n_items, ncols=0, file=sys.stdout):
        if idx >= n_items:
            break

        video_cache = video[:, :n_cache_frames].to(device)
        video_match = video[:, n_cache_frames:].to(device)
        if video_match.shape[1] == 0:
            continue
        label = label.to(device)

        # ---- Caching pass --------------------------------------------------
        model.reset()
        model.clear_counts()
        model.counting()
        model.set_mode("caching")

        with torch.inference_mode():
            _ = model(video_cache)

        all_cache_counts.append(model.total_counts())
        all_cache_block_stats.append(_collect_and_clear_block_stats(model))

        # ---- Matching pass -------------------------------------------------
        model.set_mode("matching")
        model.clear_counts()

        with torch.inference_mode():
            output = model(video_match)

        all_match_counts.append(model.total_counts())
        all_match_block_stats.append(_collect_and_clear_block_stats(model))
        model.no_counting()

        top1.update(output, label)
        top5.update(output, label)
        n_evaluated += 1

        # ---- Live progress log every 10 videos ----------------------------
        if n_evaluated % 10 == 0:
            cur_top1 = top1.compute() * 100
            cur_top5 = top5.compute() * 100
            cur_match_flops = (sum(all_match_counts) / n_evaluated).get("linear_flops", 0)
            print(
                f"[{n_evaluated:>4}/{n_items}]  "
                f"Top-1: {cur_top1:.1f}%  Top-5: {cur_top5:.1f}%  "
                f"Matching linear_flops: {cur_match_flops:.3e}",
                flush=True,
            )

    # ---- Aggregate ---------------------------------------------------------
    metrics = {"top_1": top1.compute(), "top_5": top5.compute()}
    avg_cache_counts = sum(all_cache_counts) / n_evaluated if n_evaluated else {}
    avg_match_counts = sum(all_match_counts) / n_evaluated if n_evaluated else {}

    cache_block_stats = _aggregate_caching_stats(all_cache_block_stats)
    match_block_stats = _aggregate_matching_stats(all_match_block_stats)

    # Sum per-video totals then average over videos
    total_saved_kv = (
        sum(s["total_saved_kv_flops"] for s in match_block_stats.values()) / max(n_evaluated, 1)
    )
    total_match_overhead = (
        sum(s["total_match_overhead"] for s in match_block_stats.values()) / max(n_evaluated, 1)
    )

    return {
        "metrics": metrics,
        "counts":  avg_match_counts,
        "_caching": {
            "avg_counts":     avg_cache_counts,
            "block_stats":    cache_block_stats,
            "n_cache_frames": n_cache_frames,
        },
        "_matching": {
            "avg_counts":          avg_match_counts,
            "block_stats":         match_block_stats,
            "total_saved_kv":      total_saved_kv,
            "total_match_overhead": total_match_overhead,
        },
        "_n_evaluated": n_evaluated,
    }


# ---------------------------------------------------------------------------
# Run evaluations
# ---------------------------------------------------------------------------

def run_evaluations(config, model_class, data, evaluate_function):
    device = config.get("device", get_pytorch_device())
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    model = model_class(**(config["model"]))
    model.load_state_dict(torch.load(config["weights"]))
    model = model.to(device)

    completed = []
    output_dir = Path(config["_output"])

    def do_evaluation(title):
        with open(output_dir / "output.txt", "a") as tee_file:
            model.eval()
            results = evaluate_function(device, model, data, config)

            tee_print(title, tee_file)
            tee_print(get_device_description(device), tee_file)
            tee_print("", tee_file)

            if isinstance(results, dict):
                n_evaluated  = results.pop("_n_evaluated", 0)
                caching_info = results.pop("_caching", {})
                matching_info = results.pop("_matching", {})

                tee_print(f"Videos evaluated: {n_evaluated}", tee_file)
                tee_print("", tee_file)

                _print_caching_stats(
                    caching_info.get("avg_counts", {}),
                    caching_info.get("block_stats", {}),
                    caching_info.get("n_cache_frames", 0),
                    tee_file,
                )
                _print_matching_stats(
                    matching_info.get("avg_counts", {}),
                    matching_info.get("block_stats", {}),
                    results["metrics"]["top_1"],
                    results["metrics"]["top_5"],
                    matching_info.get("total_saved_kv", 0.0),
                    matching_info.get("total_match_overhead", 0.0),
                    tee_file,
                )

                save_csv_results(results, output_dir, first_run=(len(completed) == 0))
            else:
                tee_print(results, tee_file)

            tee_print("", tee_file)
            completed.append(title)

    do_evaluation("Evaluation")


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
