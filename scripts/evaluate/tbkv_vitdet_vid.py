#!/usr/bin/env python3

import sys
import os
import random
import itertools
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.datasets.vid import VIDResize, VID
from src.models.tbkv_vitdet import TBKVViTDet
from src.utils.config import initialize_run
from src.utils.evaluate_tbkv import run_evaluations
from src.utils.misc import dict_to_device


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    return tuple(batch)


def _tensors_to_cpu(x):
    """Recursively move tensors to CPU for checkpointing."""
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, dict):
        return {k: _tensors_to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_tensors_to_cpu(v) for v in x]
    return x


def evaluate_vitdet_metrics(device, model, data, config):
    from src.tbkv.tbkv_blocks import TBKVBlock
    from src.core.base import Counts

    CHECKPOINT_EVERY = 5  # save after every N completed videos

    model.counting()

    n_items        = config.get("n_items", len(data))
    n_cache_frames = config.get("n_cache_frames", 1)
    cache_period   = config.get("cache_period", None)  # re-cache every N frames (None = only keyframes)
    max_frames     = config.get("max_frames", None)
    num_workers    = int(config.get("num_workers", 0))
    pin_memory     = bool(config.get("pin_memory", True))

    loader_kwargs = {"batch_size": 1, "collate_fn": collate_fn,
                     "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = bool(
            config.get("persistent_workers", True))

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    checkpoint_dir  = Path(config.get("_output", "results/evaluate/vitdet_vid/unknown"))
    checkpoint_path = checkpoint_dir / "checkpoint.pkl"

    # Initialise state (may be overwritten by checkpoint below).
    outputs               = []
    labels                = []
    n_frames              = 0
    n_matching_frames     = 0
    n_caching_frames      = 0
    caching_counts_acc    = defaultdict(float)
    matching_counts_acc   = defaultdict(float)
    n_cached_tokens_acc   = 0
    n_possible_tokens_acc = 0
    latency = memory = count = 0.0
    start_video = 0

    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        outputs               = ckpt["outputs"]
        labels                = ckpt["labels"]
        start_video           = ckpt["n_videos_done"]
        n_frames              = ckpt["n_frames"]
        n_matching_frames     = ckpt["n_matching_frames"]
        n_caching_frames      = ckpt["n_caching_frames"]
        caching_counts_acc    = defaultdict(float, ckpt["caching_counts_acc"])
        matching_counts_acc   = defaultdict(float, ckpt["matching_counts_acc"])
        n_cached_tokens_acc   = ckpt["n_cached_tokens_acc"]
        n_possible_tokens_acc = ckpt["n_possible_tokens_acc"]
        latency               = ckpt.get("latency", 0.0)
        memory                = ckpt.get("memory", 0.0)
        count                 = int(ckpt.get("count", 0))
        print(f"  {start_video} / {n_items} videos already done.", flush=True)

    model.clear_counts()

    # ── GPU warm-up ──────────────────────────────────────────────────────────
    cuda_timing = torch.cuda.is_available() and str(device).startswith("cuda")
    if cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        warmup_item = DataLoader(data[0], **loader_kwargs)
        model.reset()
        model.set_mode("matching")
        for frame, _ in warmup_item:
            with torch.inference_mode():
                model(frame.to(device))

    model.clear_counts()

    # ── Main loop ─────────────────────────────────────────────────────────────
    # Skip already-done videos via islice (fast: VID items are metadata-only).
    data_iter = itertools.islice(iter(data), start_video, n_items)
    pbar = tqdm(enumerate(data_iter), total=n_items,
                initial=start_video, ncols=0)

    for vid_rel_idx, vid_item in pbar:
        actual_vid_idx = start_video + vid_rel_idx
        vid_loader = DataLoader(vid_item, **loader_kwargs)
        model.reset()
        model.set_mode("matching")

        for frame_idx, (frame, annotations) in enumerate(vid_loader):
            if max_frames is not None and frame_idx >= max_frames:
                break
            n_frames += 1
            is_caching = frame_idx < n_cache_frames
            if cache_period is not None and frame_idx >= n_cache_frames:
                # Periodically refresh the cache to bound drift on long clips.
                is_caching = (frame_idx % int(cache_period) == 0)
            model.set_mode("caching" if is_caching else "matching")

            for module in model.backbone.modules():
                if hasattr(module, '_frame_stats'):
                    module._frame_stats = []
            model.clear_counts()

            with torch.inference_mode():
                if cuda_timing:
                    torch.cuda.reset_peak_memory_stats(device)
                    starter.record()
                    results = model(frame.to(device))
                    ender.record()
                    torch.cuda.synchronize()
                    latency += starter.elapsed_time(ender)
                    memory  += torch.cuda.max_memory_allocated() / (1024 * 1024)
                    count   += 1
                else:
                    results = model(frame.to(device))

            frame_counts = model.total_counts()
            if is_caching:
                n_caching_frames += 1
                for k, v in frame_counts.items():
                    caching_counts_acc[k] += float(v)
            else:
                n_matching_frames += 1
                for k, v in frame_counts.items():
                    matching_counts_acc[k] += float(v)
                for module in model.backbone.modules():
                    if isinstance(module, TBKVBlock) and module.window_size is None:
                        for stat in getattr(module, '_frame_stats', []):
                            if stat.get('phase') == 'matching':
                                n_cached_tokens_acc   += stat.get('n_bg_matched', 0)
                                n_possible_tokens_acc += stat.get('n_total', 0)

            # Store on CPU so checkpoint is always portable.
            outputs.extend(_tensors_to_cpu(results))
            labels.extend(_tensors_to_cpu(
                [dict_to_device(a, device) for a in annotations]))

        # ── Save checkpoint after each completed video ────────────────────────
        n_done = actual_vid_idx + 1
        if n_done % CHECKPOINT_EVERY == 0 or n_done == n_items:
            torch.save({
                "outputs":               outputs,
                "labels":                labels,
                "n_videos_done":         n_done,
                "n_frames":              n_frames,
                "n_matching_frames":     n_matching_frames,
                "n_caching_frames":      n_caching_frames,
                "caching_counts_acc":    dict(caching_counts_acc),
                "matching_counts_acc":   dict(matching_counts_acc),
                "n_cached_tokens_acc":   n_cached_tokens_acc,
                "n_possible_tokens_acc": n_possible_tokens_acc,
                "latency":               latency,
                "memory":                memory,
                "count":                 count,
            }, checkpoint_path)

    # ── Report ────────────────────────────────────────────────────────────────
    if count > 0:
        print(f"Latency: {latency / count} ms", flush=True)
        print(f"Memory: {memory / count} MB", flush=True)

    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    def _total_gflops(acc, n):
        return sum(v / n for v in acc.values()) / 1e9 if n > 0 else 0.0

    gflops_caching  = _total_gflops(caching_counts_acc,  n_caching_frames)
    gflops_matching = _total_gflops(matching_counts_acc, n_matching_frames)
    gflops_saved    = gflops_caching - gflops_matching

    kv_saving_pct = (
        100.0 * n_cached_tokens_acc / n_possible_tokens_acc
        if n_possible_tokens_acc > 0 else 0.0
    )
    avg_cached_per_block   = n_cached_tokens_acc   / max(n_matching_frames, 1) / 4
    avg_possible_per_block = n_possible_tokens_acc / max(n_matching_frames, 1) / 4

    def amortized(T):
        return (gflops_caching + (T - 1) * gflops_matching) / T

    print("\nTBKV Efficiency (global blocks, matching frames)", flush=True)
    print(f"  gflops_caching_frame:    {gflops_caching:.2f} GFLOPs  (≈ baseline)", flush=True)
    print(f"  gflops_matching_frame:   {gflops_matching:.2f} GFLOPs", flush=True)
    pct = 100 * gflops_saved / gflops_caching if gflops_caching else 0
    print(f"  gflops_saved_per_frame:  {gflops_saved:.2f} GFLOPs  ({pct:.1f}%)", flush=True)
    print(f"  kv_saving:               {kv_saving_pct:.1f}%  "
          f"({n_cached_tokens_acc}/{n_possible_tokens_acc} token-frames)", flush=True)
    print(f"  avg_cached_tokens/block: {avg_cached_per_block:.0f} / "
          f"{avg_possible_per_block:.0f}", flush=True)
    for T in (2, 10, 25, 100):
        print(f"  amortized_gflops @T={T:<3}: {amortized(T):.2f} GFLOPs/frame", flush=True)
    print(f"  (reference: caching-frame = {gflops_caching:.2f} GFLOPs/frame)\n", flush=True)

    # Delete checkpoint on successful completion.
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint deleted (run complete).", flush=True)

    counts = Counts()
    for k, v in matching_counts_acc.items():
        counts[k] = v / max(n_matching_frames, 1)
    return {"metrics": metrics, "counts": counts}


def main():
    torch.cuda.empty_cache()
    config = initialize_run(
        config_location=REPO_ROOT / "configs" / "evaluate" / "vitdet_vid"
    )
    set_deterministic(int(config.get("seed", 0)))
    detectron_cfg = Path(config["model"]["detectron2_config"])
    if not detectron_cfg.is_absolute():
        config["model"]["detectron2_config"] = str(REPO_ROOT / detectron_cfg)
    weights_path = Path(config["weights"])
    if not weights_path.is_absolute():
        config["weights"] = str(REPO_ROOT / weights_path)

    # Route top-level CLI overrides into the block config so that
    # e.g. `r_match=1.0` and `bg_ratio=0.7` on the command line actually take
    # effect on every TBKVBlock, not just the (unused) top-level key.
    block_cfg = config["model"]["backbone_config"]["block_config"]
    for key in ("r_match", "bg_ratio", "merging_iterations", "local_merge_ratio",
                "split_tokens", "kv_reuse_only", "caching", "raw", "matching_start_block",
                "token_skip", "secondary", "secondary_keep", "tbkv_all_blocks"):
        if key in config:
            block_cfg[key] = config[key]

    long_edge = max(config["model"]["input_shape"][-2:])
    data_root = Path(config.get("data_root", REPO_ROOT / "data" / "vid"))
    data_tar = config.get("data_tar", data_root / "data.tar")
    data_tar = None if data_tar is None else Path(data_tar)
    data = VID(
        data_root,
        split=config["split"],
        tar_path=data_tar,
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )
    run_evaluations(config, TBKVViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()
