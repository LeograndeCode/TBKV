#!/usr/bin/env python3
"""
TempoMem (VideoMAE-TBKV) ablation grid on 10% of Kinetics-400 val: cr (=
r_match, the fraction of background tokens matched-and-reused from cache)
x merge_iterations (bipartite-matching cache clustering passes, via the
same compute_merge as EventfulTBKVBlock -- see src/tbkv/tbkv_utils.py).
Named "cr" for naming consistency with the ViViT/ViTDet TempoMem grids,
even though this model calls the underlying knob r_match.

replay_matching protocol (matches videomae_full_kinetics400.py and
tbkv_videomae_kinetics400.py): the first `cache_frames` frames of clip 1
run as a discarded warmup (builds the cache; FLOPs excluded from totals),
then ALL clips are replayed in matching mode -- scored and counted.

Each video is decoded once and all grid combos are run against it before
moving to the next video (cr/merge_iterations are plain entries in
model.state, mutated between combos -- no model rebuild/reload needed).

Usage:
    python scripts/evaluate/tempomem_ablation_videomae_kinetics400.py
    python scripts/evaluate/tempomem_ablation_videomae_kinetics400.py \\
        --n_items 1988 --cr 0.25 0.5 0.75 0.95 --merge_iterations 2 4 6 8
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluate.tbkv_videomae_kinetics400 import (
    SAVED_KEY,
    computed_total,
    prepare_clips,
    top1_top5,
)
from src.datasets.kinetics400 import Kinetics400
from src.videomae_tbkv import build_videomae_tbkv


def _new_stats():
    return {"correct1": 0, "correct5": 0, "counts": {}, "cache_counts": {},
            "n_clips": 0, "n_cache_clips": 0}


def _add_counts(target, counts):
    for key, value in counts.items():
        target[key] = target.get(key, 0) + float(value)


@torch.no_grad()
def run_video(clips, label, device, model, combos, stats, cache_frames):
    clips = [clip.to(device) for clip in clips]
    n_clips = len(clips)

    for cr, mi in combos:
        key = (cr, mi)
        model.state["r_match"] = cr
        model.state["merge_iterations"] = mi

        # Caching: cache_frames-frame warmup clip (discarded, tracked
        # separately, excluded from headline totals).
        model.reset_caches()
        model.set_caching(True)
        model.clear_counts()
        model(clips[0][:, :, :cache_frames])
        _add_counts(stats[key]["cache_counts"], model.total_counts())
        stats[key]["n_cache_clips"] += 1

        # Matching: ALL clips replayed (scored + counted).
        model.set_caching(False)
        model.clear_counts()
        logits = [model(clip) for clip in clips]
        avg = torch.cat(logits, dim=0).mean(dim=0, keepdim=True)
        c1, c5, _ = top1_top5(avg, label)
        stats[key]["correct1"] += c1
        stats[key]["correct5"] += c5
        _add_counts(stats[key]["counts"], model.total_counts())
        stats[key]["n_clips"] += n_clips


def main():
    parser = argparse.ArgumentParser(
        description="TempoMem (VideoMAE-TBKV) cr x merge_iterations "
                     "ablation grid on 10% of K400"
    )
    parser.add_argument("--model", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--n_items", type=int, default=1988,
                        help="10%% of the 19,877-clip K400 val split")
    parser.add_argument("--n_clips", type=int, default=2)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--cache_frames", type=int, default=4,
                        help="warmup frames used to build the caches "
                             "(multiple of the tubelet size 2)")
    parser.add_argument("--cr", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 0.95],
                        help="r_match grid values (named cr for consistency "
                             "with the ViViT/ViTDet TempoMem grids)")
    parser.add_argument("--merge_iterations", type=int, nargs="+",
                        default=[2, 4, 6, 8])
    parser.add_argument("--merge_ratio", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=None)
    cli = parser.parse_args()

    if cli.cache_frames % 2 != 0 or not 0 < cli.cache_frames <= cli.frames_per_clip:
        parser.error("--cache_frames must be a positive multiple of 2 "
                     "no larger than --frames_per_clip")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    n_items = min(cli.n_items, len(data))

    size_tag = "tempomem_ablation" if cli.model == "vit_b" else f"tempomem_ablation_{cli.model}"
    output_dir = Path(cli.output_dir or Path(
        "results", "evaluate", "videomae_kinetics400",
        f"{size_tag}-n_items={n_items}-clips={cli.n_clips}"
        f"-cache_frames={cli.cache_frames}",
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = [(cr, mi) for cr in cli.cr for mi in cli.merge_iterations]
    stats = {c: _new_stats() for c in combos}

    model = build_videomae_tbkv(device=device, model_size=cli.model,
                                r_match=cli.cr[0],
                                merge_iterations=cli.merge_iterations[0],
                                merge_ratio=cli.merge_ratio)
    model.counting()

    print(f"Grid: {len(combos)} combos (cr x merge_iterations) over "
          f"{n_items} videos, cache_frames={cli.cache_frames}", flush=True)

    for index in range(n_items):
        video, label = data[index]
        clips = prepare_clips(video, cli.n_clips, cli.frames_per_clip)
        run_video(clips, label, device, model, combos, stats, cli.cache_frames)

        if (index + 1) % 100 == 0:
            print(f"[{index + 1}/{n_items}]", flush=True)

    # ── Report ───────────────────────────────────────────────────────────
    lines = [
        "=" * 78,
        f"TempoMem (VideoMAE-TBKV {cli.model}) cr x merge_iterations ablation, "
        f"{n_items} videos, replay_matching (cache_frames={cli.cache_frames})",
        "=" * 78,
        f"{'cr':>6} {'mi':>4} {'top1':>8} {'top5':>8} {'GFLOPs/clip':>12} "
        f"{'cache warmup G/video':>22} {'saved KV G/clip':>16}",
    ]
    summary_rows = []
    for (cr, mi), s in stats.items():
        match_flops = computed_total(s["counts"]) / max(s["n_clips"], 1)
        cache_flops = computed_total(s["cache_counts"]) / max(s["n_cache_clips"], 1)
        saved = s["counts"].get(SAVED_KEY, 0) / max(s["n_clips"], 1)
        top1 = s["correct1"] / max(n_items, 1)
        top5 = s["correct5"] / max(n_items, 1)
        lines.append(
            f"{cr:>6} {mi:>4} {top1:>8.2%} {top5:>8.2%} "
            f"{match_flops / 1e9:>12.2f} {cache_flops / 1e9:>22.2f} "
            f"{saved / 1e9:>16.2f}"
        )
        summary_rows.append({
            "cr": cr, "merge_iterations": mi, "top1": top1, "top5": top5,
            "matching_gflops_per_clip": match_flops / 1e9,
            "cache_warmup_gflops_per_video": cache_flops / 1e9,
            "saved_kv_gflops_per_clip": saved / 1e9,
        })
    report = "\n".join(lines)
    print(report, flush=True)

    with open(output_dir / "output.txt", "w") as f:
        f.write(report + "\n")
    with open(output_dir / "ablation_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"config": vars(cli), "n_items": n_items,
                   "results": summary_rows}, f, indent=2)
    print(f"\nResults saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
