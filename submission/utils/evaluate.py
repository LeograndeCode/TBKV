import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.base import dict_csv_header, dict_csv_line, dict_string
from src.utils.misc import (
    TopKAccuracy,
    get_device_description,
    get_pytorch_device,
    tee_print,
)
from .helpers import (
    _collect_and_clear_block_stats,
    _aggregate_caching_stats,
    _aggregate_matching_stats,
    save_csv_results,
    _print_caching_stats,
    _print_matching_stats,
)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_vivit_metrics(device, model, data, config):
    top1 = TopKAccuracy(k=1)
    top5 = TopKAccuracy(k=5)

    data_loader = DataLoader(data, batch_size=1, num_workers=config.get("num_workers", 2))
    n_items        = config.get("n_items", len(data_loader))
    n_cache_frames = config.get("frame_split", 4)
    # replay_matching: cache on a short warmup (a genuine, bounded clip) then
    # REPLAY the whole video in matching mode, scored and counted -- so mAP
    # covers every frame and the matching GFLOPs are the steady-state cost.
    # Default False preserves the legacy protocol (cache on first frames,
    # match on the REMAINDER), so existing results are unaffected.
    #
    # Bounded caching warmup (fixes the former ViViT accounting bug):
    # ViViT is a CLIP model. Previously the caching pass was fed a short frame
    # slice, which ViViTPreprocessing padded back up to a full temporal view and
    # fanned across temporal_views x spatial_views -- so the "warmup" silently
    # cost a full dense multi-view forward (a constant ~dense number,
    # independent of n_cache_frames). We now feed the WHOLE clip during caching
    # but (a) truncate each view to its first n_cache_frames real frames via
    # model._cache_frame_limit and (b) run spatial_only, skipping the temporal
    # head. The warmup is therefore a genuine bounded cost (~n_cache_frames/T of
    # the spatial model) and the caching + matching total is a real per-clip
    # cost. See the caching-pass block below and _forward_spatial in
    # src/models/psm_vivit.py.
    replay_matching = bool(config.get("replay_matching", False))

    all_cache_counts = []
    all_match_counts = []
    all_cache_block_stats = []
    all_match_block_stats = []
    n_evaluated = 0
    match_frames_total = 0

    for idx, (video, label) in tqdm(enumerate(data_loader), total=n_items, ncols=0, file=sys.stdout):
        if idx >= n_items:
            break
    
        # Clear cache between clips
        model.clear_cache()
        video_full = video.to(device)
        # replay_matching scores the WHOLE clip; the legacy protocol scores
        # only the frames after the caching window.
        video_match = video_full if replay_matching \
            else video[:, n_cache_frames:].to(device)
        if video_match.shape[1] == 0:
            continue
        match_frames_total += int(video_match.shape[1])
        label = label.to(device)

        # ---- Caching pass (bounded warmup) ---------------------------------
        # Feed the WHOLE clip so ViViTPreprocessing builds full-length views
        # (no padding), then _cache_frame_limit truncates each view to its
        # first n_cache_frames REAL frames, and spatial_only skips the temporal
        # head. The warmup therefore costs ~n_cache_frames/T of the spatial
        # model instead of a full dense multi-view forward. Temporal blocks are
        # left without a cache and fall back to the dense path during matching
        # (line 469 of psm_blocks.py; negligible cost, temporal is tiny).
        model.reset()
        model.clear_counts()
        model.counting()
        model.set_mode("caching")
        _prev_spatial_only = model.spatial_only
        model.spatial_only = True
        model._cache_frame_limit = n_cache_frames
        try:
            with torch.inference_mode():
                _ = model(video_full)
        finally:
            model.spatial_only = _prev_spatial_only
            model._cache_frame_limit = None

        all_cache_counts.append(model.total_counts())
        all_cache_block_stats.append(_collect_and_clear_block_stats(model))

        # ---- Matching pass -------------------------------------------------
        model.set_mode("matching")
        model.clear_counts()

        with torch.inference_mode():
            output = model(video_match)

        all_match_counts.append(model.total_counts())
        all_match_block_stats.append(_collect_and_clear_block_stats(model))
        model.no_counting()

        top1.update(output, label)
        top5.update(output, label)
        n_evaluated += 1

        # ---- Live progress log every 10 videos ----------------------------
        if n_evaluated % 10 == 0:
            cur_top1 = top1.compute() * 100
            cur_top5 = top5.compute() * 100
            cur_match_flops = (sum(all_match_counts) / n_evaluated).get("linear_flops", 0)
            print(
                f"[{n_evaluated:>4}/{n_items}]  "
                f"Top-1: {cur_top1:.1f}%  Top-5: {cur_top5:.1f}%  "
                f"Matching linear_flops: {cur_match_flops:.3e}",
                flush=True,
            )

    # ---- Aggregate ---------------------------------------------------------
    metrics = {"top_1": top1.compute(), "top_5": top5.compute()}
    avg_cache_counts = sum(all_cache_counts) / n_evaluated if n_evaluated else {}
    avg_match_counts = sum(all_match_counts) / n_evaluated if n_evaluated else {}

    cache_block_stats = _aggregate_caching_stats(all_cache_block_stats)
    match_block_stats = _aggregate_matching_stats(all_match_block_stats)

    # Sum per-video totals then average over videos
    total_saved_kv = (
        sum(s["total_saved_kv_flops"] for s in match_block_stats.values()) / max(n_evaluated, 1)
    )
    total_match_overhead = (
        sum(s["total_match_overhead"] for s in match_block_stats.values()) / max(n_evaluated, 1)
    )
    avg_matching_frames = match_frames_total / max(n_evaluated, 1)
    kv_saved_per_frame = total_saved_kv / avg_matching_frames if avg_matching_frames > 0 else 0.0
    matching_cost_per_frame = total_match_overhead / avg_matching_frames if avg_matching_frames > 0 else 0.0
    net_saved_per_frame = kv_saved_per_frame - matching_cost_per_frame

    return {
        "metrics": metrics,
        "counts":  avg_match_counts,
        "_caching": {
            "avg_counts":     avg_cache_counts,
            "block_stats":    cache_block_stats,
            "n_cache_frames": n_cache_frames,
        },
        "_matching": {
            "avg_counts":          avg_match_counts,
            "block_stats":         match_block_stats,
            "total_saved_kv":      total_saved_kv,
            "total_match_overhead": total_match_overhead,
            "avg_matching_frames": avg_matching_frames,
            "kv_saved_per_frame": kv_saved_per_frame,
            "matching_cost_per_frame": matching_cost_per_frame,
            "net_saved_per_frame": net_saved_per_frame,
        },
        "_n_evaluated": n_evaluated,
    }


# ---------------------------------------------------------------------------
# Run evaluations
# ---------------------------------------------------------------------------

def run_evaluations(config, model_class, data, evaluate_function):
    device = config.get("device", get_pytorch_device())
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    model = model_class(**(config["model"]))
    model.load_state_dict(torch.load(config["weights"]))
    model = model.to(device)

    completed = []
    output_dir = Path(config["_output"])

    def do_evaluation(title):
        with open(output_dir / "output.txt", "a") as tee_file:
            model.eval()
            results = evaluate_function(device, model, data, config)

            tee_print(title, tee_file)
            tee_print(get_device_description(device), tee_file)
            tee_print("", tee_file)

            if isinstance(results, dict):
                n_evaluated  = results.pop("_n_evaluated", 0)
                caching_info = results.pop("_caching", {})
                matching_info = results.pop("_matching", {})

                tee_print(f"Videos evaluated: {n_evaluated}", tee_file)
                tee_print("", tee_file)

                _print_caching_stats(
                    caching_info.get("avg_counts", {}),
                    caching_info.get("block_stats", {}),
                    caching_info.get("n_cache_frames", 0),
                    tee_file,
                )
                _print_matching_stats(
                    matching_info.get("avg_counts", {}),
                    matching_info.get("block_stats", {}),
                    results["metrics"]["top_1"],
                    results["metrics"]["top_5"],
                    matching_info.get("total_saved_kv", 0.0),
                    matching_info.get("total_match_overhead", 0.0),
                    matching_info.get("avg_matching_frames", 0.0),
                    matching_info.get("kv_saved_per_frame", 0.0),
                    matching_info.get("matching_cost_per_frame", 0.0),
                    matching_info.get("net_saved_per_frame", 0.0),
                    tee_file,
                )

                save_csv_results(results, output_dir, first_run=(len(completed) == 0))
            else:
                tee_print(results, tee_file)

            tee_print("", tee_file)
            completed.append(title)

    do_evaluation("Evaluation")


