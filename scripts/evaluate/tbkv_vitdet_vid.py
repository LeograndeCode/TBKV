#!/usr/bin/env python3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.datasets.vid import VIDResize, VID
from src.models.tbkv_vitdet import TBKVViTDet
from src.utils.config import initialize_run
from src.utils.evaluate_tbkv import run_evaluations
from src.utils.misc import dict_to_device, squeeze_dict


def evaluate_vitdet_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    n_cache_frames = config.get("n_cache_frames", 1)
    max_frames = config.get("max_frames", None)
    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1)
        model.reset()
        for frame_idx, (frame, annotations) in enumerate(vid_item):
            if max_frames is not None and frame_idx >= max_frames:
                break
            n_frames += 1
            model.set_mode("caching" if frame_idx < n_cache_frames else "matching")
            with torch.inference_mode():
                outputs.extend(model(frame.to(device)))
            labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def main():
    torch.cuda.empty_cache()
    config = initialize_run(
        config_location=REPO_ROOT / "configs" / "evaluate" / "vitdet_vid"
    )
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
    run_evaluations(config, TBKVViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()
