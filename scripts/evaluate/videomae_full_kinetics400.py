#!/usr/bin/env python3
"""
Full-scale VideoMAE ViT-B/16 evaluation on Kinetics-400 val: baseline, TBKV
(TempoMem) and a ToMe r-sweep in a single streaming pass.

Each video is decoded once and evaluated by every method before moving on,
so the run streams through the full 19,877-clip val split without holding
clips in memory. Progress is checkpointed so the run can resume.

Protocol per video (same as the subset scripts): 2 x 16-frame clips sampled
uniformly, center crop 224, ImageNet normalization.
- baseline: both clips at full compute (ToMe model with r=0; exact).
- tbkv:     replay_matching -- the first `cache_frames` frames of clip 1 run
  as a short warmup clip at full compute (caches built; logits discarded,
  FLOPs excluded from totals), then ALL clips (including the frames used
  for caching) are replayed in matching mode, scored and counted. Matches
  the protocol tbkv_videomae_kinetics400.py uses (2026-07-20).
- tome_r*:  both clips with r tokens merged per block.

Usage:
    python scripts/evaluate/videomae_full_kinetics400.py            # full val
    python scripts/evaluate/videomae_full_kinetics400.py --n_items 1988
"""

import argparse
import json
import sys
import time
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
from src.videomae_tbkv.videomae_tome import apply_tome


def _new_stats():
    return {"correct1": 0, "correct5": 0, "counts": {}, "n_clips": 0}


def _add_counts(stats, counts):
    for key, value in counts.items():
        stats["counts"][key] = stats["counts"].get(key, 0) + int(value)


@torch.no_grad()
def run_video(clips, label, device, model_tbkv, model_tome, tome_rs, stats,
              cache_frames):
    n_clips = len(clips)
    clips = [clip.to(device) for clip in clips]

    # Baseline: ToMe model with r=0 is the exact dense model.
    model_tome.r = 0
    model_tome.clear_counts()
    logits = torch.cat([model_tome(c) for c in clips], dim=0)
    avg = logits.mean(dim=0, keepdim=True)
    c1, c5, _ = top1_top5(avg, label)
    stats["base"]["correct1"] += c1
    stats["base"]["correct5"] += c5
    _add_counts(stats["base"], model_tome.total_counts())
    stats["base"]["n_clips"] += n_clips

    # TBKV replay_matching: cache_frames-frame warmup clip builds the
    # caches (discarded, FLOPs tracked separately under tbkv_cache and
    # excluded from the headline tbkv totals), then ALL clips are replayed
    # in matching mode -- scored and counted.
    model_tbkv.reset_caches()
    model_tbkv.set_caching(True)
    model_tbkv.clear_counts()
    model_tbkv(clips[0][:, :, :cache_frames])
    _add_counts(stats["tbkv_cache"], model_tbkv.total_counts())
    stats["tbkv_cache"]["n_clips"] += 1

    model_tbkv.set_caching(False)
    model_tbkv.clear_counts()
    logits = [model_tbkv(clip) for clip in clips]
    avg = torch.cat(logits, dim=0).mean(dim=0, keepdim=True)
    c1, c5, _ = top1_top5(avg, label)
    stats["tbkv"]["correct1"] += c1
    stats["tbkv"]["correct5"] += c5
    _add_counts(stats["tbkv"], model_tbkv.total_counts())
    stats["tbkv"]["n_clips"] += n_clips

    # ToMe r-sweep.
    for r in tome_rs:
        key = f"tome_r{r}"
        model_tome.r = r
        model_tome.clear_counts()
        logits = torch.cat([model_tome(c) for c in clips], dim=0)
        avg = logits.mean(dim=0, keepdim=True)
        c1, c5, _ = top1_top5(avg, label)
        stats[key]["correct1"] += c1
        stats[key]["correct5"] += c5
        _add_counts(stats[key], model_tome.total_counts())
        stats[key]["n_clips"] += n_clips


def main():
    parser = argparse.ArgumentParser(
        description="Full-scale VideoMAE K400: baseline + TBKV + ToMe sweep"
    )
    parser.add_argument("--model", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--n_items", type=int, default=None,
                        help="number of videos (default: full val split)")
    parser.add_argument("--n_clips", type=int, default=2)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--r_match", type=float, default=0.6)
    parser.add_argument("--cache_frames", type=int, default=4,
                        help="warmup frames used to build the caches "
                             "(multiple of the tubelet size 2); TBKV then "
                             "replays ALL clips in matching mode")
    parser.add_argument("--tome_r", type=int, nargs="+",
                        default=[33, 65, 98, 130])
    parser.add_argument("--checkpoint_every", type=int, default=250)
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
    n_items = cli.n_items if cli.n_items is not None else len(data)
    n_items = min(n_items, len(data))

    # vit_b keeps the original directory name so existing results still match.
    size_tag = "full" if cli.model == "vit_b" else f"full_{cli.model}"
    output_dir = Path(cli.output_dir or Path(
        "results", "evaluate", "videomae_kinetics400",
        f"{size_tag}-n_items={n_items}-clips={cli.n_clips}"
        f"-cache_frames={cli.cache_frames}-r_match={cli.r_match}",
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"

    methods = ["base", "tbkv"] + [f"tome_r{r}" for r in cli.tome_r]
    stats = {m: _new_stats() for m in methods + ["tbkv_cache"]}
    start = 0
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path)
        if ckpt.get("n_items") == n_items and ckpt.get("methods") == methods:
            stats = ckpt["stats"]
            start = ckpt["next_index"]
            print(f"Resuming from checkpoint at video {start}", flush=True)

    model_tbkv = build_videomae_tbkv(
        device=device, r_match=cli.r_match, model_size=cli.model
    )
    model_tbkv.counting()
    model_tome = build_videomae_tbkv(device=device, model_size=cli.model)
    apply_tome(model_tome, r=0, prop_attn=False)
    model_tome.counting()

    n_done = start
    t0 = time.time()
    for index in range(start, n_items):
        video, label = data[index]
        clips = prepare_clips(video, cli.n_clips, cli.frames_per_clip)
        run_video(clips, label, device, model_tbkv, model_tome,
                  cli.tome_r, stats, cli.cache_frames)
        n_done = index + 1

        if n_done % 50 == 0:
            b, t = stats["base"], stats["tbkv"]
            rate = (time.time() - t0) / max(n_done - start, 1)
            eta_h = rate * (n_items - n_done) / 3600
            print(
                f"[{n_done}/{n_items}] base top1={b['correct1'] / n_done:.2%} "
                f"tbkv top1={t['correct1'] / n_done:.2%} "
                f"({rate:.2f}s/video, ETA {eta_h:.1f}h)",
                flush=True,
            )
        if n_done % cli.checkpoint_every == 0:
            torch.save({"stats": stats, "next_index": n_done,
                        "n_items": n_items, "methods": methods},
                       checkpoint_path)

    # ── Report ───────────────────────────────────────────────────────────
    model_label = {"vit_b": "ViT-B/16", "vit_l": "ViT-L/16",
                   "vit_h": "ViT-H/16"}.get(cli.model, cli.model)
    lines = [
        "=" * 72,
        f"VideoMAE {model_label} "
        f"K400 val: {n_done} videos, "
        f"{cli.n_clips}x{cli.frames_per_clip} frames, "
        f"replay_matching (cache_frames={cli.cache_frames}), "
        f"r_match={cli.r_match}",
        "=" * 72,
        f"{'method':<12} {'top1':>8} {'top5':>8} {'GFLOPs/clip':>12} "
        f"{'saved KV G/clip':>16}",
    ]
    results = {}
    for m in methods:
        s = stats[m]
        flops_clip = computed_total(s["counts"]) / max(s["n_clips"], 1)
        saved = s["counts"].get(SAVED_KEY, 0) / max(s["n_clips"], 1)
        results[m] = {
            "top1": s["correct1"] / max(n_done, 1),
            "top5": s["correct5"] / max(n_done, 1),
            "flops_per_clip": flops_clip,
            "saved_kv_per_clip": saved,
            "counts_per_clip": {k: v / max(s["n_clips"], 1)
                                for k, v in s["counts"].items()},
        }
        lines.append(
            f"{m:<12} {results[m]['top1']:>8.2%} {results[m]['top5']:>8.2%} "
            f"{flops_clip / 1e9:>12.2f} {saved / 1e9:>16.2f}"
        )

    # tbkv_cache: FLOPs-only bookkeeping for the warmup pass (no predictions
    # -- excluded from the tbkv row above and from the table itself).
    cs = stats["tbkv_cache"]
    cache_flops_per_video = computed_total(cs["counts"]) / max(cs["n_clips"], 1)
    results["tbkv_cache"] = {
        "flops_per_video": cache_flops_per_video,
        "counts_per_clip": {k: v / max(cs["n_clips"], 1)
                             for k, v in cs["counts"].items()},
    }
    lines.append(
        f"(tbkv caching warmup, {cli.cache_frames} frames/video, excluded "
        f"from totals above: {cache_flops_per_video / 1e9:.2f} G/video)"
    )
    report = "\n".join(lines)
    print(report, flush=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"config": vars(cli), "n_done": n_done,
                   "results": results}, f, indent=2)
    with open(output_dir / "output.txt", "w") as f:
        f.write(report + "\n")
    if checkpoint_path.exists() and n_done == n_items:
        checkpoint_path.unlink()
    print(f"Results saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
