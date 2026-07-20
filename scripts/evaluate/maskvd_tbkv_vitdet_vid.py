#!/usr/bin/env python3
"""
Evaluate MaskVD + TBKV on ImageNet VID.

MaskVD proposes the set of unmasked tokens to recompute each frame (from the
previous frame's detections). TBKV then drops the fraction of that set which
best matches a Persistent Scene Memory (PSM) built by merging the token
embeddings of the periodic full keyframes. Dropped tokens fall back on
MaskVD's own buffer-reuse mechanism (qkv buffer `b` / output buffer
`ref_out`), so no new reuse path is needed: TBKV only ever subtracts work,
exactly as in the EventfulTBKV integration.

Runs from TBKV root. Imports the MaskVD codebase from MaskVD/ and the shared
TBKV filter from src/.

Usage:
    python scripts/evaluate/maskvd_tbkv_vitdet_vid.py [key=value ...]

Overrides (defaults match the maskvd_vitdet_vid.py baseline):
    n_items=10              Number of videos (default: 10)
    period=4                Refresh mask every N frames (default: 4)
    conf=0.5                Detection confidence threshold for mask (default: 0.5)
    margin=0                Pixel margin around detected boxes (default: 0)
    cache_reuse=0.5         Fraction of MaskVD's unmasked tokens TBKV drops
    merge_iterations=4      Bipartite-merge passes when building the PSM
    merge_ratio=0.5         Fraction of tokens merged per pass
    _output=/dev/shm/...    Output directory
"""
import importlib.util
import sys
from pathlib import Path

TBKV_ROOT = Path(__file__).resolve().parents[2]
MASKVD_ROOT = TBKV_ROOT / "MaskVD"
# NOTE: TBKV root must NOT go on sys.path — its regular `utils` package
# (with __init__.py) would shadow MaskVD's namespace-package `utils`.
# TBKV-side modules are therefore loaded standalone by file path below.
sys.path.insert(0, str(MASKVD_ROOT))

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.misc import dict_to_device, get_pytorch_device, squeeze_dict


def _load_standalone(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tbkv_keep_mask = _load_standalone(
    "tbkv_filter", TBKV_ROOT / "src" / "tbkv_filter.py"
).tbkv_keep_mask
bipartite_soft_matching = _load_standalone(
    "tbkv_merge", TBKV_ROOT / "src" / "tbkv" / "merge.py"
).bipartite_soft_matching

# The mask-proposal logic is identical to the baseline; reuse it.
get_region_mask_dynamic = _load_standalone(
    "maskvd_eval", TBKV_ROOT / "scripts" / "evaluate" / "maskvd_vitdet_vid.py"
).get_region_mask_dynamic


def merge_tokens(tok, iterations, ratio):
    """Iterated bipartite soft matching: [B, N, C] -> [B, ~N*(1-ratio)^it, C]."""
    for _ in range(iterations):
        if tok.shape[1] < 4:
            break
        m, _u = bipartite_soft_matching(tok, ratio, class_token=False, distill_token=False)
        tok = m(tok)
    return tok


class PersistentSceneMemory:
    """Merged token prototypes accumulated over the video's full keyframes."""

    def __init__(self, merge_iterations, merge_ratio, max_size=1024):
        self.merge_iterations = merge_iterations
        self.merge_ratio = merge_ratio
        self.max_size = max_size
        self.tok = None

    def update(self, frame_tokens):
        tok = frame_tokens.detach()
        if self.tok is not None:
            tok = torch.cat([self.tok, tok], dim=1)
        tok = merge_tokens(tok, self.merge_iterations, self.merge_ratio)
        # Keep the newest prototypes if the memory somehow overgrows.
        if tok.shape[1] > self.max_size:
            tok = tok[:, -self.max_size:]
        self.tok = tok


def run_evaluation(device, model, data, n_items, period=4, conf=0.5, margin=0,
                   frame_stride=1, warmup=0, cache_reuse=0.5,
                   merge_iterations=4, merge_ratio=0.5):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    total_sparsity = 0.0
    total_steps = 0
    proposed_tokens = 0
    recomputed_tokens = 0
    img_shape = [672, 672]
    n_tokens = (img_shape[0] // 16) * (img_shape[1] // 16)

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
        model.reset()
        psm = PersistentSceneMemory(merge_iterations, merge_ratio)
        frame_results = []
        step = 0
        for _raw_idx, (frame, annotations) in enumerate(loader):
            if _raw_idx % frame_stride:
                continue
            matching = step >= warmup
            if matching:
                model.counting()
                n_frames += 1
            else:
                model.no_counting()
            with torch.inference_mode():
                frame = frame.to(device)
                if cuda_timing:
                    torch.cuda.reset_peak_memory_stats(device)
                    starter.record()

                images, x_emb = model.pre_backbone(frame)
                if step % period == 0:
                    # Full keyframe: no masking; feed the PSM.
                    mask_index = None
                    sparsity = 0.0
                    psm.update(x_emb)
                else:
                    mask_index, _ = get_region_mask_dynamic(
                        frame_results,
                        image_shape=img_shape,
                        conf_threshold=conf,
                        region_size=16,
                        margin=margin,
                    )
                    mask_index = mask_index.to(device)
                    k = mask_index.shape[1]
                    proposed_tokens += k
                    if psm.tok is not None and cache_reuse > 0:
                        C = x_emb.shape[-1]
                        cand = x_emb.gather(
                            1, mask_index.unsqueeze(-1).expand(-1, -1, C)
                        )
                        keep = tbkv_keep_mask(cand, psm.tok, cache_reuse)
                        n_keep = int(keep[0].sum())
                        mask_index = mask_index[keep].reshape(1, n_keep)
                    recomputed_tokens += mask_index.shape[1]
                    sparsity = 1.0 - mask_index.shape[1] / n_tokens

                if mask_index is not None:
                    x = model.backbone(x_emb, mask_id=mask_index)
                else:
                    x = model.backbone(x_emb)
                frame_results, _ = model.post_backbone(images, x)

                if cuda_timing:
                    ender.record()
                    torch.cuda.synchronize()
                    latency += starter.elapsed_time(ender)
                    memory += torch.cuda.max_memory_allocated() / (1024 * 1024)
                    count += 1
                if matching:
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
    reuse_frac = 1.0 - recomputed_tokens / max(proposed_tokens, 1)
    return metrics, counts, avg_sparsity, reuse_frac


def main():
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

    n_items = overrides.get("n_items", 10)
    period = int(overrides.get("period", 4))
    conf = float(overrides.get("conf", 0.5))
    margin = int(overrides.get("margin", 0))
    frame_stride = int(overrides.get("frame_stride", 1))
    warmup = int(overrides.get("warmup", 0))
    cache_reuse = float(overrides.get("cache_reuse", 0.5))
    merge_iterations = int(overrides.get("merge_iterations", 4))
    merge_ratio = float(overrides.get("merge_ratio", 0.5))
    output_dir = Path(overrides.get("_output", "/dev/shm/compare/maskvd_tbkv_672_n10/"))
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
    print(f"Evaluating MaskVD+TBKV on {n_items}/{len(data)} VID videos "
          f"(period={period}, conf={conf}, margin={margin}, "
          f"cache_reuse={cache_reuse})...")

    metrics, counts, avg_sparsity, reuse_frac = run_evaluation(
        device, model, data, n_items, period=period, conf=conf, margin=margin,
        frame_stride=frame_stride, warmup=warmup, cache_reuse=cache_reuse,
        merge_iterations=merge_iterations, merge_ratio=merge_ratio,
    )

    total_gflops = sum(v for v in counts.values()) / 1e9
    lines = [
        "=== MaskVD + TBKV Results ===",
        f"n_items:         {n_items}",
        f"period:          {period}",
        f"conf:            {conf}",
        f"margin:          {margin}",
        f"cache_reuse:     {cache_reuse}",
        f"merge_iters:     {merge_iterations}  merge_ratio: {merge_ratio}",
        f"mAP@50:          {metrics['map_50']:.4f}",
        f"mAP:             {metrics['map']:.4f}",
        f"Avg sparsity:    {avg_sparsity:.3f}  (fraction of tokens SKIPPED per frame)",
        f"TBKV reuse:      {reuse_frac:.3f}  (fraction of MaskVD's unmasked tokens dropped)",
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
