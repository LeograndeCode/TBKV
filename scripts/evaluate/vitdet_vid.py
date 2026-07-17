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
from src.models.vitdet import ViTDet
from src.utils.config import initialize_run
from src.utils.evaluate import run_evaluations
from src.utils.misc import dict_to_device, squeeze_dict


def evaluate_vitdet_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))

    # ── Latency / peak-memory harness (identical protocol to TBKV script) ─────
    cuda_timing = torch.cuda.is_available() and str(device).startswith("cuda")
    latency = memory = count = 0.0
    if cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        # GPU warm-up on the first video so kernels are compiled/cached.
        model.reset()
        for frame, _ in DataLoader(data[0], batch_size=1):
            with torch.inference_mode():
                model(frame.to(device))
        model.clear_counts()

    # frame_stride > 1 keeps only every k-th frame, which widens the temporal
    # gap between consecutive model inputs without changing the content. This
    # is the stress axis for methods that assume frame-to-frame adjacency.
    frame_stride = int(config.get("frame_stride", 1))
    # warmup > 0: skip the first N frames of every video from BOTH scoring and
    # FLOP counting (matching-frame-only protocol, shared with EventfulTBKV).
    warmup = int(config.get("warmup", 0))
    model.clear_counts()

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1)
        model.reset()
        step = 0
        for f_i, (frame, annotations) in enumerate(vid_item):
            if f_i % frame_stride:
                continue
            matching = step >= warmup
            step += 1
            if matching:
                model.counting()
                n_frames += 1
            else:
                model.no_counting()
            frame = frame.to(device)
            with torch.inference_mode():
                if cuda_timing:
                    torch.cuda.reset_peak_memory_stats(device)
                    starter.record()
                    results = model(frame)
                    ender.record()
                    torch.cuda.synchronize()
                    latency += starter.elapsed_time(ender)
                    memory += torch.cuda.max_memory_allocated() / (1024 * 1024)
                    count += 1
                else:
                    results = model(frame)
            if matching:
                outputs.extend(results)
                labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

    if count > 0:
        print(f"Latency: {latency / count} ms", flush=True)
        print(f"Memory: {memory / count} MB", flush=True)

    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def main():
    
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
    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()
