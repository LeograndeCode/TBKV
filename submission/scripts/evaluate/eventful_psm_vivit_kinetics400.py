#!/usr/bin/env python3
"""Evaluate EventfulPSM on ViViT / Kinetics-400.

Default two-pass protocol: the first ``frame_split`` frames of every clip are
run in caching mode (to build the K/V cache), and the REMAINING frames are run
in matching mode (where eventful tokens try to reuse the cache). Both passes
are scored/counted, since together they cover the whole clip.

With ``replay_matching=true``: the first ``frame_split`` frames build the
cache in a discarded, uncounted caching pass, then the matching pass replays
the FULL clip (all frames, from frame 0) with cache reuse active -- scored
and counted. GFLOPs are reported for the matching pass only (the caching
warmup is excluded), matching the analogous VID protocol.
"""

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.kinetics400 import Kinetics400
from src.models.eventful_psm_vivit import EventfulPSMViViT
from src.utils.config import initialize_run
from src.utils.misc import TopKAccuracy, get_pytorch_device, set_policies
from src.core.policies import TokenNormTopK
from utils.evaluate import run_evaluations


def evaluate_eventful_psm(device, model, data, config):
    # The block subclasses the real EventfulBlock, whose token gates need a
    # selection policy assigned (token_top_k), exactly as the stock Eventful
    # evaluation does.
    for k in config.get("token_top_k", []):
        set_policies(model, TokenNormTopK, k=k)
    top1 = TopKAccuracy(k=1)
    top5 = TopKAccuracy(k=5)

    loader = DataLoader(data, batch_size=1, num_workers=config.get("num_workers", 2))
    n_items = config.get("n_items", len(loader))
    n_cache_frames = config.get("frame_split", 4)
    # frame_stride > 1 keeps only every k-th frame of the clip before the
    # caching/matching split, widening the temporal gap between consecutive
    # inputs -- the stress axis for adjacency-based reuse.
    frame_stride = int(config.get("frame_stride", 1))
    # replay_matching: caching pass is discarded/uncounted, matching pass
    # replays the FULL clip (see module docstring).
    replay_matching = bool(config.get("replay_matching", False))

    n_evaluated = 0
    cache_flops = {}
    match_flops = {}
    for idx, (video, label) in tqdm(enumerate(loader), total=n_items, ncols=0, file=sys.stdout):
        if idx >= n_items:
            break

        if frame_stride > 1:
            video = video[:, ::frame_stride]
        video_cache = video[:, :n_cache_frames].to(device)
        video_match = video.to(device) if replay_matching else video[:, n_cache_frames:].to(device)
        if video_match.shape[1] == 0:
            continue
        label = label.to(device)

        # Fresh state per clip.
        model.reset()
        model.clear_cache()

        # Caching pass (output ignored) then matching pass (scored). The cache
        # built during caching must survive into matching, so there is NO reset
        # between the two passes.
        model.set_mode("caching")
        model.clear_counts()
        if replay_matching:
            # Caching is a discarded warmup: not counted, matching replays
            # the whole clip so its cost alone is the honest steady-state.
            model.no_counting()
        else:
            # Caching covers frames the matching pass never revisits, so both
            # passes together are needed to cover the whole clip's cost.
            model.counting()
        with torch.inference_mode():
            _ = model(video_cache)
        model.no_counting()
        for k, v in model.total_counts().items():
            cache_flops[k] = cache_flops.get(k, 0) + float(v)

        model.set_mode("matching")
        model.clear_counts(); model.counting()
        with torch.inference_mode():
            output = model(video_match)
        model.no_counting()
        for k, v in model.total_counts().items():
            match_flops[k] = match_flops.get(k, 0) + float(v)

        top1.update(output, label)
        top5.update(output, label)
        n_evaluated += 1

        if n_evaluated % 5 == 0:
            print(f"[{n_evaluated:>4}/{n_items}]  "
                  f"Top-1: {top1.compute() * 100:.1f}%  "
                  f"Top-5: {top5.compute() * 100:.1f}%", flush=True)

    gc = sum(cache_flops.values()) / max(n_evaluated, 1) / 1e9
    gm = sum(match_flops.values()) / max(n_evaluated, 1) / 1e9
    if replay_matching:
        return (
            "EventfulPSM — ViViT / Kinetics-400 (replay_matching)\n"
            f"  Videos evaluated : {n_evaluated}\n"
            f"  Caching frames   : {n_cache_frames} (discarded, uncounted)   "
            f"frame_stride: {frame_stride}\n"
            f"  Top-1 Accuracy   : {top1.compute() * 100:.2f}%\n"
            f"  Top-5 Accuracy   : {top5.compute() * 100:.2f}%\n"
            f"  GFLOPs/clip      : matching {gm:.0f} (caching warmup, "
            f"excluded: {gc:.0f})"
        )
    return (
        "EventfulPSM — ViViT / Kinetics-400\n"
        f"  Videos evaluated : {n_evaluated}\n"
        f"  Caching frames   : {n_cache_frames}   frame_stride: {frame_stride}\n"
        f"  Top-1 Accuracy   : {top1.compute() * 100:.2f}%\n"
        f"  Top-5 Accuracy   : {top5.compute() * 100:.2f}%\n"
        f"  GFLOPs/clip      : caching {gc:.0f} + matching {gm:.0f} = TOTAL {gc + gm:.0f}"
    )


_METRIC_PATTERNS = {
    "top1": r"Top-1 Accuracy\s*:\s*([\d.]+)%",
    "top5": r"Top-5 Accuracy\s*:\s*([\d.]+)%",
    "matching_gflops_per_clip": r"matching\s*([\d.]+)\s*\(caching",
}


def _parse_report_metrics(report):
    values = {}
    for key, pattern in _METRIC_PATTERNS.items():
        m = re.search(pattern, report)
        values[key] = float(m.group(1)) if m else None
    return values


def _psm_blocks(model):
    for m in model.modules():
        if hasattr(m, "cache_reuse") and hasattr(m, "merge_iterations"):
            yield m


def run_ablation_sweep(device, model, data, config, cache_reuse_values, merge_iterations_values):
    """
    Sweep cache_reuse x merge_iterations in-process (both are plain runtime
    attributes on each EventfulPSMBlock, so no rebuild/reload is needed
    between combos -- see src/psm/blocks.py). Writes every combo's
    full report to output.txt and a compact ablation_summary.csv.
    """
    output_dir = Path(config["_output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = [(cr, mi) for cr in cache_reuse_values for mi in merge_iterations_values]
    summary_rows = []
    with open(output_dir / "output.txt", "w") as f:
        for done, (cr, mi) in enumerate(combos, start=1):
            for block in _psm_blocks(model):
                block.cache_reuse = float(cr)
                block.merge_iterations = int(mi)
            header = f"[{done}/{len(combos)}] cache_reuse={cr} merge_iterations={mi}"
            print(header, flush=True)
            report = evaluate_eventful_psm(device, model, data, config)
            print(report, flush=True)
            f.write(header + "\n" + report + "\n\n")
            f.flush()

            row = {"cache_reuse": cr, "merge_iterations": mi}
            row.update(_parse_report_metrics(report))
            summary_rows.append(row)

    summary_path = output_dir / "ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[ablation] Summary written to {summary_path}", flush=True)


def main():
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_kinetics400")
    )
    spatial_cfg = config["model"].get("spatial_config", {}).get("block_config")

    cache_reuse_cfg = config.get("cache_reuse")
    merge_iterations_cfg = config.get("merge_iterations")
    sweep = isinstance(cache_reuse_cfg, list) or isinstance(merge_iterations_cfg, list)

    data = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )

    if sweep:
        # cache_reuse / merge_iterations are swept in-process (see
        # run_ablation_sweep); seed the block config with placeholder
        # scalars so model construction succeeds, then route the other,
        # non-swept overrides normally.
        cr_values = cache_reuse_cfg if isinstance(cache_reuse_cfg, list) else [cache_reuse_cfg]
        mi_values = merge_iterations_cfg if isinstance(merge_iterations_cfg, list) else [merge_iterations_cfg]
        if spatial_cfg is not None:
            spatial_cfg["cache_reuse"] = cr_values[0]
            spatial_cfg["merge_iterations"] = mi_values[0]
            for key in ("token_keep", "merge_ratio", "caching", "substitute"):
                if key in config:
                    spatial_cfg[key] = config[key]

        device = config.get("device", get_pytorch_device())
        model = EventfulPSMViViT(**config["model"])
        model.load_state_dict(torch.load(config["weights"], map_location=device))
        model = model.to(device).eval()

        run_ablation_sweep(device, model, data, config, cr_values, mi_values)
        return

    # Single-combo path (unchanged): route CLI overrides into the spatial
    # block config and run through the standard harness.
    if spatial_cfg is not None:
        for key in ("token_keep", "merge_iterations", "merge_ratio",
                    "cache_reuse", "caching", "substitute"):
            if key in config:
                spatial_cfg[key] = config[key]
    # frame_stride is read from config directly by the eval loop; nothing to route.

    run_evaluations(config, EventfulPSMViViT, data, evaluate_eventful_psm)


if __name__ == "__main__":
    main()
