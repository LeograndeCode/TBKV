#!/usr/bin/env python3
"""Scene-cut stress test for Eventful+TBKV (TempoMem) on ImageNet-VID / ViTDet.

Simulates a hard scene cut: the model warms up (Eventful gate state + TBKV
content cache) on the first `warmup` frames of video A, then must process
video B. Three paired arms, all scored on the SAME frames of B (index >=
warmup):

  control      warm-up on B itself (the normal protocol; upper bound)
  cut_stale    warm-up on A, no refresh (cache stays stale for all of B)
  cut_refresh  warm-up on A, periodic refresh every cache_period frames
               (the paper's claim: degradation bounded by the refresh period)

Usage:
  python scripts/evaluate/scene_cut_vitdet.py tbkv_eventful_filter_672 \
      "token_top_k=[512]" cache_reuse=0.5 n_pairs=10 cache_period=16
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.datasets.vid import VIDResize, VID
from src.models.vitdet import ViTDet
from src.utils.config import initialize_run
from src.utils.evaluate import run_evaluations
from src.utils.misc import dict_to_device, squeeze_dict

from scripts.evaluate.tbkv_eventful_vitdet_vid import (
    _clear_caches,
    _set_caching,
)

ARMS = ("control", "cut_stale", "cut_refresh")


def evaluate_scene_cut(device, model, data, config):
    warmup = int(config.get("warmup", 4))
    cache_period = int(config.get("cache_period", 16))
    n_pairs = int(config.get("n_pairs", 10))
    max_frames = config.get("max_frames", None)

    model.no_counting()  # accuracy stress test; FLOPs are not the question here

    outputs = {arm: [] for arm in ARMS}
    labels = {arm: [] for arm in ARMS}

    for pair_idx in tqdm(range(n_pairs), ncols=0):
        vid_a = data[2 * pair_idx]
        vid_b = data[2 * pair_idx + 1]
        frames_a = [(f, a) for f, a in DataLoader(vid_a, batch_size=1)]
        frames_b = [(f, a) for f, a in DataLoader(vid_b, batch_size=1)]
        if max_frames is not None:
            frames_b = frames_b[: int(max_frames)]
        if len(frames_a) < warmup or len(frames_b) <= warmup:
            continue

        for arm in ARMS:
            model.reset()
            warm_source = frames_b if arm == "control" else frames_a
            _set_caching(model, True)
            with torch.inference_mode():
                for frame, _ in warm_source[:warmup]:
                    model(frame.to(device))
            _set_caching(model, False)

            # All arms are scored on the identical frame set: B[warmup:].
            for step, (frame, annotations) in enumerate(frames_b[warmup:],
                                                        start=warmup):
                refresh = (arm == "cut_refresh") and (step % cache_period == 0)
                if refresh:
                    _clear_caches(model)
                _set_caching(model, refresh)
                with torch.inference_mode():
                    results = model(frame.to(device))
                outputs[arm].extend(results)
                labels[arm].append(
                    squeeze_dict(dict_to_device(annotations, device), dim=0)
                )

    all_metrics = {}
    for arm in ARMS:
        mean_ap = MeanAveragePrecision()
        mean_ap.update(outputs[arm], labels[arm])
        metrics = mean_ap.compute()
        all_metrics[f"{arm}_map50"] = float(metrics["map_50"])
        all_metrics[f"{arm}_map"] = float(metrics["map"])

    print("\nScene-cut stress test "
          f"(warmup={warmup}, cache_period={cache_period}, "
          f"n_pairs={n_pairs}):", flush=True)
    for arm in ARMS:
        print(f"  {arm:<12} mAP@50 = {all_metrics[f'{arm}_map50']:.4f}  "
              f"mAP = {all_metrics[f'{arm}_map']:.4f}", flush=True)

    return {"metrics": all_metrics}


def main():
    config = initialize_run(
        config_location=REPO_ROOT / "configs" / "evaluate" / "vitdet_vid"
    )
    bc = config["model"].get("backbone_config", {}).get("block_config")
    if bc is not None:
        for key in ("cache_reuse", "merge_iterations", "merge_ratio", "substitute"):
            if key in config:
                bc[key] = config[key]

    detectron_cfg = Path(config["model"]["detectron2_config"])
    if not detectron_cfg.is_absolute():
        config["model"]["detectron2_config"] = str(REPO_ROOT / detectron_cfg)
    weights_path = Path(config["weights"])
    if not weights_path.is_absolute():
        config["weights"] = str(REPO_ROOT / weights_path)

    long_edge = max(config["model"]["input_shape"][-2:])
    data = VID(
        REPO_ROOT / "data" / "vid",
        split=config["split"],
        tar_path=REPO_ROOT / "data" / "vid" / "data.tar",
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )
    run_evaluations(config, ViTDet, data, evaluate_scene_cut)


if __name__ == "__main__":
    main()
