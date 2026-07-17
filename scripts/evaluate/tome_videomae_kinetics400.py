#!/usr/bin/env python3
"""
Evaluate ToMe (Token Merging, Bolya et al.) on VideoMAE ViT-B/16 (official
Kinetics-400 fine-tune, weights/videomae_vit_b.pth) on the K400 val split.

Same data protocol as the TBKV evaluation (tbkv_videomae_kinetics400.py):
32 frames sampled uniformly per video -> two 16-frame clips (8*14*14 = 1568
tokens each); prediction = average of the clip logits. Unlike TBKV, ToMe
has no caching/matching phases -- every clip runs with r tokens merged per
block -- so r=0 is the exact baseline and each r > 0 is one operating point.

FLOPs are measured with src.core.counting, directly comparable with the
TBKV numbers.

Usage:
    python scripts/evaluate/tome_videomae_kinetics400.py --n_items 25 --r 0 33 65 98
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluate.tbkv_videomae_kinetics400 import (
    accumulate,
    computed_total,
    prepare_clips,
    top1_top5,
)
from src.datasets.kinetics400 import Kinetics400
from src.videomae_tbkv import build_videomae_tbkv
from src.videomae_tbkv.videomae_tome import apply_tome


@torch.no_grad()
def evaluate_tome(model, clip_sets, labels, device, r):
    model.r = r
    tag = f"tome r={r}" if r > 0 else "baseline"
    correct1 = correct5 = 0
    counts_sum = {}
    n_clips_total = 0
    t0 = time.time()
    for i, (clips, label) in enumerate(zip(clip_sets, labels)):
        model.clear_counts()
        logits = [model(clip.to(device)) for clip in clips]
        avg_logits = torch.cat(logits, dim=0).mean(dim=0, keepdim=True)
        c1, c5, _ = top1_top5(avg_logits, label)
        correct1 += c1
        correct5 += c5
        accumulate(counts_sum, model.total_counts())
        n_clips_total += len(clips)
        print(
            f"  [{tag}] video {i + 1}/{len(clip_sets)} "
            f"top1={correct1 / (i + 1):.2%} top5={correct5 / (i + 1):.2%}",
            flush=True,
        )
    n = len(clip_sets)
    return {
        "r": r,
        "top1": correct1 / n,
        "top5": correct5 / n,
        "flops_per_clip": computed_total(counts_sum) / n_clips_total,
        "flops_per_video": computed_total(counts_sum) / n,
        "counts_per_clip": {k: v / n_clips_total for k, v in counts_sum.items()},
        "runtime_s": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ToMe on VideoMAE ViT-B (Kinetics-400)"
    )
    parser.add_argument("--n_items", type=int, default=25)
    parser.add_argument("--n_clips", type=int, default=2)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument(
        "--r", type=int, nargs="+", default=[0, 65],
        help="Tokens merged per block; 0 = exact baseline. "
             "Multiple values evaluate multiple operating points.",
    )
    parser.add_argument(
        "--prop_attn", action="store_true",
        help="Proportional attention (ToMe recommends off for MAE models).",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    r_values = sorted(set(cli.r))
    output_dir = cli.output_dir or str(
        Path(
            "results",
            "evaluate",
            "videomae_kinetics400",
            f"tome-n_items={cli.n_items}-clips={cli.n_clips}"
            f"-r={','.join(str(r) for r in r_values)}",
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data: first n_items of the (seed-shuffled) K400 val split.
    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    print(f"Kinetics-400 val: {len(data)} videos, using first {cli.n_items}")
    clip_sets, labels = [], []
    for index in range(cli.n_items):
        video, label = data[index]
        clip_sets.append(prepare_clips(video, cli.n_clips, cli.frames_per_clip))
        labels.append(label)

    # Model: official VideoMAE ViT-B/16 K400 fine-tune, patched with ToMe.
    model = build_videomae_tbkv(device=device)
    apply_tome(model, r=0, prop_attn=cli.prop_attn)
    model.counting()

    runs = []
    for r in r_values:
        print("\n" + "=" * 60)
        print(f"ToMe r={r}" + (" (baseline)" if r == 0 else "")
              + (f", prop_attn={cli.prop_attn}" if r > 0 else ""))
        print("=" * 60)
        runs.append(evaluate_tome(model, clip_sets, labels, device, r))

    baseline = next((run for run in runs if run["r"] == 0), None)

    lines = [
        "",
        "=" * 60,
        "SUMMARY",
        "=" * 60,
        f"Videos: {cli.n_items} | clips/video: {cli.n_clips} "
        f"x {cli.frames_per_clip} frames | prop_attn: {cli.prop_attn}",
        "",
        f"{'r':>6} {'top1':>8} {'top5':>8} {'GFLOPs/clip':>12} {'reduction':>10}",
    ]
    for run in runs:
        reduction = (
            f"{1 - run['flops_per_clip'] / baseline['flops_per_clip']:.1%}"
            if baseline is not None else "n/a"
        )
        lines.append(
            f"{run['r']:>6} {run['top1']:>8.2%} {run['top5']:>8.2%} "
            f"{run['flops_per_clip'] / 1e9:>12.2f} {reduction:>10}"
        )
    print("\n".join(lines))

    results = {
        "config": {
            "n_items": cli.n_items,
            "n_clips": cli.n_clips,
            "frames_per_clip": cli.frames_per_clip,
            "r_values": r_values,
            "prop_attn": cli.prop_attn,
            "weights": "weights/videomae_vit_b.pth",
            "model": "VideoMAE ViT-B/16 (official K400 fine-tune) + ToMe",
        },
        "runs": runs,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "output.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
