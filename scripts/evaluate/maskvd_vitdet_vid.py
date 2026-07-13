#!/usr/bin/env python3
"""
Evaluate MaskVD dynamic token masking on ImageNet VID.
Runs from TBKV root. Imports MaskVD codebase from MaskVD/.

Usage:
    python scripts/evaluate/maskvd_vitdet_vid.py [key=value ...]

Overrides:
    n_items=10              Number of videos (default: all)
    period=4                Refresh mask every N frames (default: 4)
    conf=0.5                Detection confidence threshold for mask (default: 0.5)
    margin=0                Pixel margin around detected boxes (default: 0)
    _output=/dev/shm/...    Output directory (default: /dev/shm/compare/maskvd_672/)
"""
import sys
from pathlib import Path

TBKV_ROOT = Path(__file__).resolve().parents[2]
MASKVD_ROOT = TBKV_ROOT / "MaskVD"
sys.path.insert(0, str(MASKVD_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.misc import dict_to_device, get_pytorch_device, squeeze_dict


def get_region_mask_dynamic(results, image_shape, conf_threshold=0.5, region_size=16, margin=0):
    """Build a token mask from previous-frame detection results.

    Returns:
        mask_index: shape (1, K) tensor of kept token indices
        sparsity:   fraction of tokens that are SKIPPED
    """
    mask = torch.zeros(image_shape)
    has_box = False
    for result in results:
        for i, bbox in enumerate(result["boxes"]):
            if result["scores"][i] > conf_threshold:
                has_box = True
                x1, y1, x2, y2 = bbox
                y1m = max(0, int(y1) - margin)
                y2m = min(image_shape[0], int(y2) + margin)
                x1m = max(0, int(x1) - margin)
                x2m = min(image_shape[1], int(x2) + margin)
                mask[y1m:y2m, x1m:x2m] = 1
    if not has_box:
        # No confident detections — keep all tokens (no skip)
        mask = torch.ones(image_shape)
    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight, stride=region_size)
    mask_index = torch.nonzero(y.flatten() > 0).reshape(1, -1)
    sparsity = 1.0 - mask_index.shape[1] / y.numel()
    return mask_index, sparsity


def run_evaluation(device, model, data, n_items, period=4, conf=0.5, margin=0):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    total_sparsity = 0.0
    total_steps = 0
    img_shape = [672, 672]  # global attention input size

    # ── Latency / peak-memory harness (identical protocol to TBKV script) ─────
    cuda_timing = torch.cuda.is_available() and str(device).startswith("cuda")
    latency = memory = count = 0.0
    if cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        model.reset()
        for frame, _ in DataLoader(data[0], batch_size=1):
            with torch.inference_mode():
                model(frame.to(device), None)
        model.clear_counts()

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        loader = DataLoader(vid_item, batch_size=1)
        n_frames += len(loader)
        model.reset()
        frame_results = []
        step = 0
        for frame, annotations in loader:
            with torch.inference_mode():
                if step % period == 0:
                    # Full inference: no masking
                    mask_index = None
                    sparsity = 0.0
                else:
                    mask_index, sparsity = get_region_mask_dynamic(
                        frame_results,
                        image_shape=img_shape,
                        conf_threshold=conf,
                        region_size=16,
                        margin=margin,
                    )
                    mask_index = mask_index.to(device)

                frame = frame.to(device)
                if cuda_timing:
                    torch.cuda.reset_peak_memory_stats(device)
                    starter.record()
                    frame_results, _ = model(frame, mask_index)
                    ender.record()
                    torch.cuda.synchronize()
                    latency += starter.elapsed_time(ender)
                    memory += torch.cuda.max_memory_allocated() / (1024 * 1024)
                    count += 1
                else:
                    frame_results, _ = model(frame, mask_index)
                outputs.extend(frame_results)
                labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

            total_sparsity += sparsity
            total_steps += 1
            step += 1

    if count > 0:
        print(f"Latency: {latency / count} ms", flush=True)
        print(f"Memory: {memory / count} MB", flush=True)

    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    counts = model.total_counts() / n_frames
    model.clear_counts()
    avg_sparsity = total_sparsity / max(total_steps, 1)
    return metrics, counts, avg_sparsity


def main():
    # Parse key=value overrides from CLI
    overrides = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try:
                overrides[k] = int(v)
            except ValueError:
                try:
                    overrides[k] = float(v)
                except ValueError:
                    overrides[k] = v
        else:
            print(f"Warning: ignoring non-override argument: {arg}")

    n_items = overrides.get("n_items", None)
    period = int(overrides.get("period", 4))
    conf = float(overrides.get("conf", 0.5))
    margin = int(overrides.get("margin", 0))
    output_dir = Path(overrides.get("_output", "/dev/shm/compare/maskvd_672/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_pytorch_device()
    weights_path = TBKV_ROOT / "weights" / "vitdet_b_vid.pth"
    data_root = TBKV_ROOT / "data" / "vid"
    detectron2_config = str(TBKV_ROOT / "configs" / "detectron" / "vitdet_b_vid.py")

    model_kwargs = dict(
        classes=30,
        detectron2_config=detectron2_config,
        input_shape=[3, 672, 672],
        normalize_mean=[123.675, 116.28, 103.53],
        normalize_std=[58.395, 57.12, 57.375],
        output_channels=256,
        patch_size=[16, 16],
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        mask=True,
        backbone_config=dict(
            backbone="windowed",
            depth=12,
            position_encoding_size=[14, 14],
            window_indices=[0, 1, 3, 4, 6, 7, 9, 10],
            block_config=dict(
                dim=768,
                relative_embedding_size=[64, 64],
                heads=12,
                mlp_ratio=4,
                window_size=[14, 14],
            ),
        ),
    )

    print("Loading MaskVD model...")
    model = ViTDet(**model_kwargs)
    msg = model.load_state_dict(
        torch.load(str(weights_path), map_location="cpu"), strict=False
    )
    print(f"Weight loading: {msg}")
    model = model.to(device)
    model.eval()

    data = VID(
        data_root,
        split="vid_val",
        tar_path=data_root / "data.tar",
        combined_transform=VIDResize(short_edge_length=420, max_size=672),
    )

    if n_items is None:
        n_items = len(data)
    print(f"Evaluating MaskVD on {n_items}/{len(data)} VID videos "
          f"(period={period}, conf={conf}, margin={margin})...")

    metrics, counts, avg_sparsity = run_evaluation(
        device, model, data, n_items, period=period, conf=conf, margin=margin
    )

    total_gflops = sum(v for v in counts.values()) / 1e9
    lines = [
        "=== MaskVD Results ===",
        f"n_items:         {n_items}",
        f"period:          {period}",
        f"conf:            {conf}",
        f"margin:          {margin}",
        f"mAP@50:          {metrics['map_50']:.4f}",
        f"mAP:             {metrics['map']:.4f}",
        f"Avg sparsity:    {avg_sparsity:.3f}  (fraction of tokens SKIPPED per frame)",
        f"GFLOPs/frame:    {total_gflops:.3f}",
    ]
    for line in lines:
        print(line)

    with open(output_dir / "results.txt", "w") as f:
        for line in lines:
            f.write(line + "\n")
        f.write(f"\nCounts: {dict(counts)}\n")

    print(f"\nResults saved to {output_dir}/results.txt")


if __name__ == "__main__":
    main()
