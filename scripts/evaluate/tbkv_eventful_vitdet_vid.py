#!/usr/bin/env python3
"""EventfulTBKV (real Eventful + recompute-set filter) on ImageNet-VID / ViTDet.

Same as the stock Eventful ViTDet eval, but two-phase per video: the first
`warmup` frames run in caching mode (Eventful builds its state AND TBKV builds
its content cache -- these frames are NOT scored or counted), then the remaining
frames run in matching mode, where TBKV drops cache-matched tokens from
Eventful's recompute set. mAP and per-frame FLOPs are reported over the matching
frames only, which is the honest steady-state cost once warm-up has amortised.
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


def _tbkv_blocks(model):
    for m in model.modules():
        if hasattr(m, "cache_reuse") and hasattr(m, "caching"):
            yield m


def _set_caching(model, flag):
    for b in _tbkv_blocks(model):
        b.caching = flag


def evaluate_eventful_tbkv_vitdet(device, model, data, config):
    model.counting(); model.clear_counts()
    warmup = int(config.get("warmup", 4))
    frame_stride = int(config.get("frame_stride", 1))
    n_items = config.get("n_items", len(data))

    outputs, labels = [], []
    match_frames = 0

    # Count ONCE across the whole run: clear the global accumulator here, then
    # gate counting per frame (warm-up frames off, matching frames on). Calling
    # clear_counts() inside the loop would zero every prior video's counts.
    model.clear_counts()

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        loader = DataLoader(vid_item, batch_size=1)
        model.reset()
        _set_caching(model, True)
        step = 0
        for f_i, (frame, annotations) in enumerate(loader):
            if f_i % frame_stride:
                continue
            frame = frame.to(device)
            if step == warmup:
                _set_caching(model, False)   # warm-up done: build cache, start matching
            # Warm-up FLOPs are invisible (streaming): count matching frames only.
            if step >= warmup:
                model.counting()
            else:
                model.no_counting()
            with torch.inference_mode():
                results = model(frame)
            if step >= warmup:
                outputs.extend(results)
                labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
                match_frames += 1
            step += 1

    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    counts = model.total_counts() / max(match_frames, 1)
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def main():
    config = initialize_run(
        config_location=REPO_ROOT / "configs" / "evaluate" / "vitdet_vid"
    )
    # Route TBKV overrides into the block config.
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
    run_evaluations(config, ViTDet, data, evaluate_eventful_tbkv_vitdet)


if __name__ == "__main__":
    main()
