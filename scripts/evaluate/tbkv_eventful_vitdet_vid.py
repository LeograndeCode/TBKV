#!/usr/bin/env python3
"""EventfulTBKV (real Eventful + recompute-set filter) on ImageNet-VID / ViTDet.

Same as the stock Eventful ViTDet eval, but two-phase per video: the first
`warmup` frames run in caching mode (Eventful builds its state AND TBKV builds
its content cache -- these frames are NOT scored or counted), then the remaining
frames run in matching mode, where TBKV drops cache-matched tokens from
Eventful's recompute set. mAP and per-frame FLOPs are reported over the matching
frames only, which is the honest steady-state cost once warm-up has amortised.
"""
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

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


def _clear_caches(model):
    """Drop each block's content cache so the next caching frame rebuilds it.

    _finalize_cache() only fires when cache is None, so a periodic-refresh
    frame must clear the stale cache (and any leftover accumulator) first.
    """
    for b in _tbkv_blocks(model):
        if hasattr(b, "cache"):
            b.cache = None
        if hasattr(b, "_acc"):
            b._acc = []


def evaluate_eventful_tbkv_vitdet(device, model, data, config):
    model.counting(); model.clear_counts()
    warmup = int(config.get("warmup", 4))
    frame_stride = int(config.get("frame_stride", 1))
    n_items = config.get("n_items", len(data))
    # Optional extensions (all default-off, preserving the original protocol):
    # cache_period: re-enter caching mode for one frame every P matching frames
    #   (the paper's periodic refresh). Refresh frames are scored but excluded
    #   from the matching-frame FLOPs average, like warm-up frames.
    # whole_stream: count and score ALL frames (warm-up/refresh included) and
    #   average FLOPs over all of them (the SOTA-table accounting).
    # measure_latency: CUDA-event wall-clock and peak memory per frame, under
    #   the same warm-up harness as scripts/evaluate/tbkv_vitdet_vid.py.
    cache_period = config.get("cache_period", None)
    whole_stream = bool(config.get("whole_stream", False))
    measure_latency = bool(config.get("measure_latency", False))
    # replay_matching: build the cache on the first `warmup` frames, then
    # REPLAY the video from frame 0 entirely in matching mode, scoring and
    # counting every frame. This matches the accounting of methods that are
    # evaluated at matching cost on all frames (dense/STGT/MaskVD rows):
    # mAP covers the full video, GFLOPs/frame averages matching frames only
    # (the caching pass is neither scored nor counted).
    replay_matching = bool(config.get("replay_matching", False))

    outputs, labels = [], []
    match_frames = 0
    total_frames = 0
    refresh_frames = 0
    latency = memory = 0.0
    timed = 0

    cuda_timing = measure_latency and torch.cuda.is_available()
    if cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        # GPU warm-up pass so the first timed frame is not paying init costs.
        warm_loader = DataLoader(data[0], batch_size=1)
        model.reset()
        _set_caching(model, False)
        for frame, _ in warm_loader:
            with torch.inference_mode():
                model(frame.to(device))
            break

    # Count ONCE across the whole run: clear the global accumulator here, then
    # gate counting per frame (warm-up frames off, matching frames on). Calling
    # clear_counts() inside the loop would zero every prior video's counts.
    model.clear_counts()

    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        loader = DataLoader(vid_item, batch_size=1)
        model.reset()
        _set_caching(model, True)

        if replay_matching:
            # Caching pass: first `warmup` frames build gate state + cache.
            model.no_counting()
            step = 0
            with torch.inference_mode():
                for f_i, (frame, _) in enumerate(loader):
                    if f_i % frame_stride:
                        continue
                    model(frame.to(device))
                    step += 1
                    if step >= warmup:
                        break
            _set_caching(model, False)
            model.counting()
            # Matching pass: every frame, from frame 0, scored and counted.
            for f_i, (frame, annotations) in enumerate(DataLoader(vid_item, batch_size=1)):
                if f_i % frame_stride:
                    continue
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
                        timed += 1
                    else:
                        results = model(frame)
                total_frames += 1
                match_frames += 1
                outputs.extend(results)
                labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
            continue

        step = 0
        for f_i, (frame, annotations) in enumerate(loader):
            if f_i % frame_stride:
                continue
            frame = frame.to(device)
            is_caching = step < warmup
            if cache_period is not None and step >= warmup:
                # Periodic refresh: one caching frame every cache_period frames.
                is_caching = (step % int(cache_period)) == 0
                if is_caching:
                    refresh_frames += 1
                    _clear_caches(model)  # stale cache dropped; rebuilt this frame
            _set_caching(model, is_caching)
            counted = whole_stream or not is_caching
            if counted:
                model.counting()
            else:
                model.no_counting()
            with torch.inference_mode():
                if cuda_timing:
                    torch.cuda.reset_peak_memory_stats(device)
                    starter.record()
                    results = model(frame)
                    ender.record()
                    torch.cuda.synchronize()
                    latency += starter.elapsed_time(ender)
                    memory += torch.cuda.max_memory_allocated() / (1024 * 1024)
                    timed += 1
                else:
                    results = model(frame)
            total_frames += 1
            scored = whole_stream or step >= warmup
            if scored:
                outputs.extend(results)
                labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
            if counted and not is_caching:
                match_frames += 1
            step += 1

    if cuda_timing and timed > 0:
        print(f"Latency: {latency / timed} ms", flush=True)
        print(f"Memory: {memory / timed} MB", flush=True)
    if cache_period is not None:
        print(f"Refresh frames (dense cost, excluded from matching avg): "
              f"{refresh_frames}", flush=True)

    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    denom = total_frames if whole_stream else match_frames
    counts = model.total_counts() / max(denom, 1)
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def main():
    config = initialize_run(
        config_location=REPO_ROOT / "configs" / "evaluate" / "vitdet_vid"
    )
    # A list-valued cache_reuse sweeps one full evaluation per value, each into
    # its own "<output>-cache_reuse=<value>/" directory (same naming a CLI
    # override would produce). A scalar keeps the single-run behavior.
    cr_sweep = config.get("cache_reuse")
    if not isinstance(cr_sweep, (list, tuple)):
        cr_sweep = None

    # Route TBKV overrides into the block config.
    bc = config["model"].get("backbone_config", {}).get("block_config")
    if bc is not None:
        for key in ("cache_reuse", "merge_iterations", "merge_ratio", "substitute"):
            if key in config and not (key == "cache_reuse" and cr_sweep):
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
    if cr_sweep is None:
        run_evaluations(config, ViTDet, data, evaluate_eventful_tbkv_vitdet)
        return

    base_output = config["_output"].rstrip("/")
    for cr in cr_sweep:
        run_config = copy.deepcopy(config)
        run_config["cache_reuse"] = cr
        run_config["model"]["backbone_config"]["block_config"]["cache_reuse"] = cr
        run_config["_output"] = f"{base_output}-cache_reuse={cr}/"
        output_dir = Path(run_config["_output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(run_config, output_dir / "config.yml", resolve=True)
        print(f"########## cache_reuse={cr}", flush=True)
        run_evaluations(run_config, ViTDet, data, evaluate_eventful_tbkv_vitdet)


if __name__ == "__main__":
    main()
