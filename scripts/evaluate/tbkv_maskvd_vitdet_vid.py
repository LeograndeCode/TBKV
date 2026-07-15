#!/usr/bin/env python3
"""MaskVD + TBKV on ImageNet-VID / ViTDet.

Exactly MaskVD, with TBKV bolted on as a filter over MaskVD's token mask -- the
MaskVD ViTDet model is imported and run unchanged. This mirrors the Eventful+TBKV
integration: the base method proposes an active token set, and TBKV drops the
fraction of it that matches a content cache.

MaskVD proposes `mask_index` (the tokens inside the previous frame's detection
regions). On each keyframe (mask_index is None -> full frame) we refresh a cache
of per-patch content descriptors. On masked frames we drop the `cache_reuse`
fraction of mask_index tokens that best match that cache, so the model processes
fewer tokens. Nothing in MaskVD's model changes; the whole integration is this
one filter in the eval loop.

Usage:
    python scripts/evaluate/tbkv_maskvd_vitdet_vid.py n_items=10 cache_reuse=0.5
"""
import sys
from pathlib import Path

TBKV_ROOT = Path(__file__).resolve().parents[2]
MASKVD_ROOT = TBKV_ROOT / "MaskVD"
# Only MaskVD on the path -- TBKV's own utils/ would otherwise shadow MaskVD's.
sys.path.insert(0, str(MASKVD_ROOT))

import importlib.util

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.misc import dict_to_device, get_pytorch_device, squeeze_dict

# Load the shared TBKV filter by file path, so we don't add TBKV_ROOT to
# sys.path (which would shadow MaskVD's utils/datasets/models packages).
_spec = importlib.util.spec_from_file_location(
    "tbkv_filter", str(TBKV_ROOT / "src" / "tbkv_filter.py")
)
_tbkv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tbkv)
tbkv_keep_mask = _tbkv.tbkv_keep_mask

REGION = 16
IMG = 672
GRID = IMG // REGION  # 42 -> 1764 tokens


def get_region_mask_dynamic(results, image_shape, conf_threshold=0.5, region_size=16, margin=0):
    """MaskVD's mask: keep tokens inside the previous frame's detection boxes.

    Copied from scripts/evaluate/maskvd_vitdet_vid.py so this script depends only
    on MaskVD's own modules (importing that script would re-add TBKV to the path).
    """
    mask = torch.zeros(image_shape)
    has_box = False
    for result in results:
        for i, bbox in enumerate(result["boxes"]):
            if result["scores"][i] > conf_threshold:
                has_box = True
                x1, y1, x2, y2 = bbox
                y1m = max(0, int(y1) - margin); y2m = min(image_shape[0], int(y2) + margin)
                x1m = max(0, int(x1) - margin); x2m = min(image_shape[1], int(x2) + margin)
                mask[y1m:y2m, x1m:x2m] = 1
    if not has_box:
        mask = torch.ones(image_shape)
    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight, stride=region_size)
    mask_index = torch.nonzero(y.flatten() > 0).reshape(1, -1)
    sparsity = 1.0 - mask_index.shape[1] / y.numel()
    return mask_index, sparsity


def patch_descriptors(frame, device):
    """Per-token content descriptor: the raw 16x16x3 patch, on the 42x42 grid."""
    f = F.interpolate(frame.to(device).float(), size=(IMG, IMG), mode="bilinear", align_corners=False)
    # [1, 3, 672, 672] -> [1, 3*16*16, 1764] -> [1, 1764, 768]
    patches = F.unfold(f, kernel_size=REGION, stride=REGION)
    return patches.transpose(1, 2).contiguous()


def run_evaluation(device, model, data, n_items, period=4, conf=0.5, margin=0,
                   frame_stride=1, cache_reuse=0.5):
    model.counting(); model.clear_counts()
    n_frames = 0
    outputs, labels = [], []
    total_sparsity = total_steps = 0
    img_shape = [IMG, IMG]

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        loader = DataLoader(vid_item, batch_size=1)
        model.reset()
        frame_results = []
        step = 0
        cache = None   # per-clip content cache, refreshed on keyframes
        for _raw_idx, (frame, annotations) in enumerate(loader):
            if _raw_idx % frame_stride:
                continue
            n_frames += 1
            with torch.inference_mode():
                desc = patch_descriptors(frame, device)     # [1, 1764, 768]
                if step % period == 0:
                    # Keyframe: full inference, and refresh the TBKV cache.
                    mask_index = None
                    sparsity = 0.0
                    cache = desc
                else:
                    mask_index, sparsity = get_region_mask_dynamic(
                        frame_results, image_shape=img_shape,
                        conf_threshold=conf, region_size=REGION, margin=margin,
                    )
                    mask_index = mask_index.to(device)
                    # ---- TBKV: drop cache-matched tokens from MaskVD's mask ----
                    if cache is not None and mask_index.shape[1] > 0:
                        cand = desc.gather(
                            1, mask_index.unsqueeze(-1).expand(-1, -1, desc.shape[-1])
                        )
                        keep = tbkv_keep_mask(cand, cache, cache_reuse)  # [1, K]
                        mask_index = mask_index[keep].unsqueeze(0)
                        sparsity = 1.0 - mask_index.shape[1] / (GRID * GRID)

                frame = frame.to(device)
                frame_results, _ = model(frame, mask_index)
                outputs.extend(frame_results)
                labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

            total_sparsity += sparsity
            total_steps += 1
            step += 1

    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    counts = model.total_counts()
    return metrics, counts, total_sparsity / max(total_steps, 1), n_frames


def main():
    overrides = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            for cast in (int, float):
                try:
                    v = cast(v); break
                except ValueError:
                    pass
            overrides[k] = v

    n_items = overrides.get("n_items", None)
    period = int(overrides.get("period", 4))
    conf = float(overrides.get("conf", 0.5))
    margin = int(overrides.get("margin", 0))
    frame_stride = int(overrides.get("frame_stride", 1))
    cache_reuse = float(overrides.get("cache_reuse", 0.5))

    device = get_pytorch_device()
    weights_path = TBKV_ROOT / "weights" / "vitdet_b_vid.pth"
    data_root = TBKV_ROOT / "data" / "vid"
    detectron2_config = str(TBKV_ROOT / "configs" / "detectron" / "vitdet_b_vid.py")

    model_kwargs = dict(
        classes=30, detectron2_config=detectron2_config, input_shape=[3, 672, 672],
        normalize_mean=[123.675, 116.28, 103.53], normalize_std=[58.395, 57.12, 57.375],
        output_channels=256, patch_size=[16, 16], scale_factors=[4.0, 2.0, 1.0, 0.5],
        mask=True,
        backbone_config=dict(
            backbone="windowed", depth=12, position_encoding_size=[14, 14],
            window_indices=[0, 1, 3, 4, 6, 7, 9, 10],
            block_config=dict(dim=768, relative_embedding_size=[64, 64], heads=12,
                              mlp_ratio=4, window_size=[14, 14]),
        ),
    )
    print("Loading MaskVD model...")
    model = ViTDet(**model_kwargs)
    msg = model.load_state_dict(torch.load(str(weights_path), map_location="cpu"), strict=False)
    print(f"Weight loading: {msg}")
    model = model.to(device).eval()

    data = VID(data_root, split="vid_val", tar_path=data_root / "data.tar",
               combined_transform=VIDResize(short_edge_length=420, max_size=672))
    if n_items is None:
        n_items = len(data)
    print(f"MaskVD+TBKV on {n_items} videos (period={period}, cache_reuse={cache_reuse})...")

    metrics, counts, sparsity, n_frames = run_evaluation(
        device, model, data, n_items, period=period, conf=conf, margin=margin,
        frame_stride=frame_stride, cache_reuse=cache_reuse,
    )
    gflops_per_frame = sum(float(v) for v in counts.values()) / 1e9 / max(n_frames, 1)
    print("=== MaskVD+TBKV Results ===")
    print(f"  cache_reuse:     {cache_reuse}")
    print(f"  mAP:             {metrics['map']:.4f}")
    print(f"  mAP@50:          {metrics['map_50']:.4f}")
    print(f"  avg sparsity:    {sparsity:.3f}  (fraction of tokens skipped)")
    print(f"  GFLOPs/frame:    {gflops_per_frame:.3f}")


if __name__ == "__main__":
    main()
