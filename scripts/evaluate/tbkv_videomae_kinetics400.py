#!/usr/bin/env python3
"""
Evaluate TBKV on VideoMAE ViT-B/16 (official Kinetics-400 fine-tune,
weights/videomae_vit_b.pth) on the Kinetics-400 val split.

VideoMAE consumes 16-frame clips with joint space-time attention (8*14*14 =
1568 tokens), so TBKV operates at clip granularity:
- 32 frames are sampled uniformly per video and split into two 16-frame clips.
- Baseline: both clips at full compute; prediction = average of clip logits.
- TBKV: the first cache_frames frames (default 4, must be a multiple of the
  tubelet size 2) run in caching mode as a short warmup clip (full compute;
  per-block background tokens and their aligned K/V are cached; logits
  discarded), then ALL clips run in matching mode, reusing cached K/V for
  background tokens that match the cache.
  Prediction = average of the matched clips' logits.

FLOPs are measured with src.core.counting (linear/matmul/conv/bias/add),
consistent with the other evaluations in this repo. The headline TBKV
numbers count ONLY the matching passes; the caching warmup is reported
separately. K/V projections saved by cache hits are reported under
"saved_kv_linear_flops" and excluded from totals.

Usage:
    python scripts/evaluate/tbkv_videomae_kinetics400.py --n_items 25
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.kinetics400 import Kinetics400
from src.videomae_tbkv import build_videomae_tbkv

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

SAVED_KEY = "saved_kv_linear_flops"


def prepare_clips(video, n_clips, frames_per_clip):
    """
    video: [T, C, H, W] uint8 tensor (decoded at short-edge 224).
    Samples n_clips*frames_per_clip frames uniformly, center-crops to
    224x224, normalizes, and returns a list of [1, 3, T, 224, 224] clips.
    """
    n_frames = n_clips * frames_per_clip
    total = video.shape[0]
    if total >= n_frames:
        indices = torch.linspace(0, total - 1, n_frames).long()
    else:
        indices = torch.arange(total).repeat((n_frames // total) + 1)[:n_frames]
    frames = video[indices].float() / 255.0
    frames = TF.center_crop(frames, [224, 224])
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    clips = frames.reshape(n_clips, frames_per_clip, *frames.shape[1:])
    return [clip.permute(1, 0, 2, 3).unsqueeze(0) for clip in clips]


def top1_top5(logits, label):
    top5 = logits.topk(5, dim=-1).indices.squeeze(0)
    pred = top5[0].item()
    return pred == label, label in top5.tolist(), pred


def computed_total(counts):
    """Total computed FLOPs from a Counts dict (saved KV flops excluded)."""
    return sum(v for k, v in counts.items() if k != SAVED_KEY)


def accumulate(target, counts):
    for key, value in counts.items():
        target[key] = target.get(key, 0) + value


@torch.no_grad()
def evaluate_baseline(model, clip_sets, labels, device):
    correct1 = correct5 = 0
    counts_sum = {}
    n_clips_total = 0
    t0 = time.time()
    model.reset_caches()
    model.set_caching(False)
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
            f"  [baseline] video {i + 1}/{len(clip_sets)} "
            f"top1={correct1 / (i + 1):.2%} top5={correct5 / (i + 1):.2%}",
            flush=True,
        )
    n = len(clip_sets)
    return {
        "top1": correct1 / n,
        "top5": correct5 / n,
        "flops_per_clip": computed_total(counts_sum) / n_clips_total,
        "flops_per_video": computed_total(counts_sum) / n,
        "counts_per_clip": {k: v / n_clips_total for k, v in counts_sum.items()},
        "runtime_s": time.time() - t0,
    }


@torch.no_grad()
def evaluate_tbkv(model, clip_sets, labels, device, cache_frames):
    correct1 = correct5 = 0
    cache_counts_sum = {}
    match_counts_sum = {}
    n_cache_clips = 0
    n_match_clips = 0
    t0 = time.time()
    for i, (clips, label) in enumerate(zip(clip_sets, labels)):
        model.reset_caches()

        # CACHING PHASE: the first cache_frames frames as a short warmup
        # clip at full compute build the per-block caches; logits discarded.
        model.set_caching(True)
        model.clear_counts()
        model(clips[0][:, :, :cache_frames].to(device))
        accumulate(cache_counts_sum, model.total_counts())
        n_cache_clips += 1

        # MATCHING PHASE: every clip (including the frames used for
        # caching) reuses cached K/V.
        model.set_caching(False)
        model.clear_counts()
        logits = [model(clip.to(device)) for clip in clips]
        accumulate(match_counts_sum, model.total_counts())
        n_match_clips += len(clips)

        avg_logits = torch.cat(logits, dim=0).mean(dim=0, keepdim=True)
        c1, c5, _ = top1_top5(avg_logits, label)
        correct1 += c1
        correct5 += c5
        print(
            f"  [tbkv] video {i + 1}/{len(clip_sets)} "
            f"top1={correct1 / (i + 1):.2%} top5={correct5 / (i + 1):.2%}",
            flush=True,
        )
    n = len(clip_sets)
    cache_total = computed_total(cache_counts_sum)
    match_total = computed_total(match_counts_sum)
    return {
        "top1": correct1 / n,
        "top5": correct5 / n,
        "cache_warmup_flops_per_video": cache_total / n_cache_clips,
        "matching_flops_per_clip": match_total / n_match_clips,
        # Matching only; the caching warmup is excluded by design.
        "flops_per_video": match_total / n,
        "saved_kv_flops_per_matching_clip": match_counts_sum.get(SAVED_KEY, 0)
        / n_match_clips,
        "cache_counts_per_clip": {
            k: v / n_cache_clips for k, v in cache_counts_sum.items()
        },
        "matching_counts_per_clip": {
            k: v / n_match_clips for k, v in match_counts_sum.items()
        },
        "runtime_s": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TBKV on VideoMAE ViT-B (Kinetics-400)"
    )
    parser.add_argument("--n_items", type=int, default=25)
    parser.add_argument("--n_clips", type=int, default=2)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--cache_frames", type=int, default=4,
                        help="warmup frames used to build the caches "
                             "(multiple of the tubelet size 2)")
    parser.add_argument("--r_match", type=float, default=0.6)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    cli = parser.parse_args()

    if cli.cache_frames % 2 != 0 or not 0 < cli.cache_frames <= cli.frames_per_clip:
        parser.error("--cache_frames must be a positive multiple of 2 "
                     "no larger than --frames_per_clip")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = cli.output_dir or str(
        Path(
            "results",
            "evaluate",
            "videomae_kinetics400",
            f"tbkv-n_items={cli.n_items}-clips={cli.n_clips}"
            f"-cache_frames={cli.cache_frames}-r_match={cli.r_match}",
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

    # Model: official VideoMAE ViT-B/16 K400 fine-tune with TBKV built in.
    model = build_videomae_tbkv(
        device=device, r_match=cli.r_match, verbose=cli.verbose
    )
    model.counting()

    print("\n" + "=" * 60)
    print("BASELINE (VideoMAE, full compute on all clips)")
    print("=" * 60)
    baseline = evaluate_baseline(model, clip_sets, labels, device)

    print("\n" + "=" * 60)
    print(f"TBKV ({cli.cache_frames} caching frames + {cli.n_clips} "
          f"matching clip(s), r_match={cli.r_match})")
    print("=" * 60)
    tbkv = evaluate_tbkv(model, clip_sets, labels, device, cli.cache_frames)

    flop_reduction_video = 1 - tbkv["flops_per_video"] / baseline["flops_per_video"]
    flop_reduction_matching = (
        1 - tbkv["matching_flops_per_clip"] / baseline["flops_per_clip"]
    )

    lines = [
        "",
        "=" * 60,
        "SUMMARY",
        "=" * 60,
        f"Videos: {cli.n_items} | clips/video: {cli.n_clips} "
        f"x {cli.frames_per_clip} frames | caching frames: {cli.cache_frames} "
        f"| r_match: {cli.r_match}",
        "",
        f"Baseline top1: {baseline['top1']:.2%}  top5: {baseline['top5']:.2%}",
        f"TBKV     top1: {tbkv['top1']:.2%}  top5: {tbkv['top5']:.2%}",
        "",
        f"Baseline FLOPs/clip: {baseline['flops_per_clip'] / 1e9:.2f} G",
        f"TBKV matching FLOPs/clip: {tbkv['matching_flops_per_clip'] / 1e9:.2f} G",
        f"Baseline FLOPs/video: {baseline['flops_per_video'] / 1e9:.2f} G",
        f"TBKV matching FLOPs/video: {tbkv['flops_per_video'] / 1e9:.2f} G",
        f"(caching warmup, excluded from totals: "
        f"{tbkv['cache_warmup_flops_per_video'] / 1e9:.2f} G/video)",
        "",
        f"FLOP reduction (matching clips vs baseline): {flop_reduction_matching:.1%}",
        f"FLOP reduction (whole video, matching only): {flop_reduction_video:.1%}",
        "",
        f"KV linear FLOPs saved per matching clip: "
        f"{tbkv['saved_kv_flops_per_matching_clip'] / 1e9:.2f} G",
        f"Baseline runtime: {baseline['runtime_s']:.1f}s | "
        f"TBKV runtime: {tbkv['runtime_s']:.1f}s",
    ]
    print("\n".join(lines))

    results = {
        "config": {
            "n_items": cli.n_items,
            "n_clips": cli.n_clips,
            "frames_per_clip": cli.frames_per_clip,
            "cache_frames": cli.cache_frames,
            "r_match": cli.r_match,
            "weights": "weights/videomae_vit_b.pth",
            "model": "VideoMAE ViT-B/16 (official K400 fine-tune)",
        },
        "baseline": baseline,
        "tbkv": tbkv,
        "flop_reduction_matching": flop_reduction_matching,
        "flop_reduction_video": flop_reduction_video,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "output.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
