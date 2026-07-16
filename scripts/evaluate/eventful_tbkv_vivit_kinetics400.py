#!/usr/bin/env python3
"""Evaluate EventfulTBKV on ViViT / Kinetics-400.

Same two-pass protocol as TBKV: the first ``frame_split`` frames of every clip
are run in caching mode (to build the K/V cache), and the remaining frames are
run in matching mode (where eventful tokens try to reuse the cache). Only the
matching-pass output is scored.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.kinetics400 import Kinetics400
from src.models.eventful_tbkv_vivit import EventfulTBKVViViT
from src.utils.config import initialize_run
from src.utils.misc import TopKAccuracy, set_policies
from src.core.policies import TokenNormTopK
from utils.evaluate import run_evaluations


def evaluate_eventful_tbkv(device, model, data, config):
    # The block subclasses the real EventfulBlock, whose token gates need a
    # selection policy assigned (token_top_k), exactly as the stock Eventful
    # evaluation does.
    for k in config.get("token_top_k", []):
        set_policies(model, TokenNormTopK, k=k)
    top1 = TopKAccuracy(k=1)
    top5 = TopKAccuracy(k=5)

    loader = DataLoader(data, batch_size=1, num_workers=config.get("num_workers", 2))
    n_items = config.get("n_items", len(loader))
    n_cache_frames = config.get("frame_split", 4)
    # frame_stride > 1 keeps only every k-th frame of the clip before the
    # caching/matching split, widening the temporal gap between consecutive
    # inputs -- the stress axis for adjacency-based reuse.
    frame_stride = int(config.get("frame_stride", 1))

    n_evaluated = 0
    cache_flops = {}
    match_flops = {}
    for idx, (video, label) in tqdm(enumerate(loader), total=n_items, ncols=0, file=sys.stdout):
        if idx >= n_items:
            break

        if frame_stride > 1:
            video = video[:, ::frame_stride]
        video_cache = video[:, :n_cache_frames].to(device)
        video_match = video[:, n_cache_frames:].to(device)
        if video_match.shape[1] == 0:
            continue
        label = label.to(device)

        # Fresh state per clip.
        model.reset()
        model.clear_cache()

        # Caching pass (output ignored) then matching pass (scored). The cache
        # built during caching must survive into matching, so there is NO reset
        # between the two passes. FLOPs are counted for BOTH passes: the caching
        # pass is a full ViViT forward (ViViT resamples any input to fixed size),
        # so reporting only the matching pass would badly understate cost.
        model.set_mode("caching")
        model.clear_counts(); model.counting()
        with torch.inference_mode():
            _ = model(video_cache)
        model.no_counting()
        for k, v in model.total_counts().items():
            cache_flops[k] = cache_flops.get(k, 0) + float(v)

        model.set_mode("matching")
        model.clear_counts(); model.counting()
        with torch.inference_mode():
            output = model(video_match)
        model.no_counting()
        for k, v in model.total_counts().items():
            match_flops[k] = match_flops.get(k, 0) + float(v)

        top1.update(output, label)
        top5.update(output, label)
        n_evaluated += 1

        if n_evaluated % 5 == 0:
            print(f"[{n_evaluated:>4}/{n_items}]  "
                  f"Top-1: {top1.compute() * 100:.1f}%  "
                  f"Top-5: {top5.compute() * 100:.1f}%", flush=True)

    gc = sum(cache_flops.values()) / max(n_evaluated, 1) / 1e9
    gm = sum(match_flops.values()) / max(n_evaluated, 1) / 1e9
    return (
        "EventfulTBKV — ViViT / Kinetics-400\n"
        f"  Videos evaluated : {n_evaluated}\n"
        f"  Caching frames   : {n_cache_frames}   frame_stride: {frame_stride}\n"
        f"  Top-1 Accuracy   : {top1.compute() * 100:.2f}%\n"
        f"  Top-5 Accuracy   : {top5.compute() * 100:.2f}%\n"
        f"  GFLOPs/clip      : caching {gc:.0f} + matching {gm:.0f} = TOTAL {gc + gm:.0f}\n"
        f"  (Eventful temporal_24 reference, whole-clip: 613 GFLOPs, Top-1 71%)"
    )


def main():
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_kinetics400")
    )
    # Route CLI overrides into the spatial block config.
    spatial_cfg = config["model"].get("spatial_config", {}).get("block_config")
    if spatial_cfg is not None:
        for key in ("token_keep", "merge_iterations", "merge_ratio",
                    "tbkv_tau", "cache_reuse", "caching", "substitute"):
            if key in config:
                spatial_cfg[key] = config[key]
    # frame_stride is read from config directly by the eval loop; nothing to route.

    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    run_evaluations(config, EventfulTBKVViViT, data, evaluate_eventful_tbkv)


if __name__ == "__main__":
    main()
