from pathlib import Path
import json

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


# ---------------------------------------------------------------------------
# Per-block stats helpers
# ---------------------------------------------------------------------------

def _collect_block_stats(model):
    """Return {block_name: [frame-stat dicts]} and reset each block's list."""
    try:
        from src.tbkv.tbkv_blocks import TBKVBlock
    except ModuleNotFoundError:
        from eventful_transformer.tbkv_blocks import TBKVBlock
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, TBKVBlock):
            out[name] = list(getattr(module, '_frame_stats', []))
            module._frame_stats = []
    return out


def _mean_dicts(dicts):
    if not dicts:
        return {}
    keys = [k for k, v in dicts[0].items() if isinstance(v, (int, float))]
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def _aggregate_block_stats(all_video_stats):
    """
    all_video_stats: list (one per video) of {block_name: [frame-stat dicts]}
    Returns {block_name: {caching_avg_per_frame: dict, matching_avg_per_frame: dict}}
    """
    if not all_video_stats:
        return {}
    block_names = list(all_video_stats[0].keys())
    result = {}
    for name in block_names:
        caching, matching = [], []
        for vid in all_video_stats:
            for entry in vid.get(name, []):
                (caching if entry['phase'] == 'caching' else matching).append(entry)
        result[name] = {}
        if caching:
            result[name]['caching_avg_per_frame'] = _mean_dicts(caching)
        if matching:
            result[name]['matching_avg_per_frame'] = _mean_dicts(matching)
    return result


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

N_VIDEOS = 1000

def evaluate_vivit_metrics(device, model, data, config):
    top_1 = TopKAccuracy(k=1)
    top_5 = TopKAccuracy(k=5)

    data_loader = DataLoader(data, batch_size=1)
    n_videos       = config.get("n_items", N_VIDEOS)
    n_cache_frames = config.get("n_cache_frames", 16)

    all_counts = []
    all_video_block_stats = []
    use_cuda_amp = (hasattr(device, "type") and device.type == "cuda")

    for idx, (video, label) in tqdm(enumerate(data_loader), total=n_videos, ncols=0):
        if idx >= n_videos:
            break

        torch.cuda.empty_cache()
        video_cache = video[:, :n_cache_frames].to(device)
        video_match = video[:, n_cache_frames:].to(device)

        # Skip videos that have no remaining frames for matching
        if video_match.shape[1] == 0:
            continue
        label = label.to(device)

        # ----------------------------------------------------------------
        # Caching pass
        # ----------------------------------------------------------------
        model.reset()
        model.clear_counts()
        model.counting()

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=use_cuda_amp):
                _ = model(video_cache)

        cache_counts = model.total_counts()

        # ----------------------------------------------------------------
        # Matching pass
        # ----------------------------------------------------------------
        model.set_mode("matching")
        model.clear_counts()

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=use_cuda_amp):
                output = model(video_match)

        match_counts = model.total_counts()
        model.no_counting()

        total_counts = cache_counts + match_counts
        all_counts.append({
            "caching":  cache_counts,
            "matching": match_counts,
            "total":    total_counts,
        })

        all_video_block_stats.append(_collect_block_stats(model))

        top_1.update(output, label)
        top_5.update(output, label)

    metrics = {"top_1": top_1.compute(), "top_5": top_5.compute()}

    n_evaluated = len(all_counts)
    if n_evaluated > 0:
        avg_caching  = sum(c["caching"]  for c in all_counts) / n_evaluated
        avg_matching = sum(c["matching"] for c in all_counts) / n_evaluated
        avg_total    = sum(c["total"]    for c in all_counts) / n_evaluated
        print("\n--- FLOPs breakdown (per-video average) ---")
        print(f"  Caching pass:  {dict_string(avg_caching)}")
        print(f"  Matching pass: {dict_string(avg_matching)}")
        print(f"  Total:         {dict_string(avg_total)}")
        print("---\n")
        counts = avg_matching
    else:
        counts = {}
        avg_caching = avg_matching = avg_total = {}

    model.clear_counts()
    return {
        "metrics": metrics,
        "counts":  counts,
        "_block_stats":   all_video_block_stats,
        "_flops_summary": {
            "caching":  avg_caching,
            "matching": avg_matching,
            "total":    avg_total,
        },
        "_n_evaluated": n_evaluated,
    }


# ---------------------------------------------------------------------------
# Save statistics.json
# ---------------------------------------------------------------------------

def _save_statistics(results, output_dir):
    block_stats   = results.pop("_block_stats",   [])
    flops_summary = results.pop("_flops_summary", {})
    n_evaluated   = results.pop("_n_evaluated",   0)

    block_agg = _aggregate_block_stats(block_stats)

    total_saved_kv = 0.0
    total_overhead = 0.0
    for bdata in block_agg.values():
        m = bdata.get('matching_avg_per_frame', {})
        total_saved_kv += m.get('saved_kv_linear_flops',   0.0)
        total_overhead += m.get('matching_overhead_flops',  0.0)

    def _c2d(c):
        return {k: float(v) for k, v in c.items()} if c else {}

    stats = {
        "n_videos_evaluated": n_evaluated,
        "flops_per_video": {
            "caching_pass":  _c2d(flops_summary.get("caching",  {})),
            "matching_pass": _c2d(flops_summary.get("matching", {})),
            "total":         _c2d(flops_summary.get("total",    {})),
        },
        "matching_savings_per_matching_frame": {
            "description":             "summed over all transformer blocks",
            "saved_kv_linear_flops":   total_saved_kv,
            "matching_overhead_flops": total_overhead,
            "net_savings":             total_saved_kv - total_overhead,
        },
        "per_block": block_agg,
    }

    out_path = output_dir / "statistics.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {out_path}")


# ---------------------------------------------------------------------------
# run_evaluations
# ---------------------------------------------------------------------------

def run_evaluations(config, model_class, data, evaluate_function):
    device = config.get("device", get_pytorch_device())
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    torch.cuda.empty_cache()
    model = model_class(**(config["model"]))
    sd = torch.load(config["weights"], map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Filter out known acceptable mismatches
    real_missing = [k for k in missing if "qkv" not in k]
    if real_missing:
        print(f"Warning: missing keys in checkpoint: {real_missing}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected}")
    model = model.to(device)

    completed  = []
    output_dir = Path(config["_output"])

    def do_evaluation(title):
        with open(output_dir / "output.txt", "a") as tee_file:
            model.eval()
            results = evaluate_function(device, model, data, config)

            _save_statistics(results, output_dir)

            tee_print(title, tee_file)
            tee_print(get_device_description(device), tee_file)
            if isinstance(results, dict):
                save_csv_results(results, output_dir, first_run=(len(completed) == 0))
                for key, val in results.items():
                    tee_print(key.capitalize(), tee_file)
                    tee_print(dict_string(val), tee_file)
            else:
                tee_print(results, tee_file)
            tee_print("", tee_file)
            completed.append(title)

    do_evaluation("Evaluation")


def save_csv_results(results, output_dir, first_run=False):
    for key, val in results.items():
        with open(output_dir / f"{key}.csv", "a") as csv_file:
            if first_run:
                print(dict_csv_header(val), file=csv_file)
            print(dict_csv_line(val), file=csv_file)
