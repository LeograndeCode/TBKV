#!/usr/bin/env python3
"""
Plot per-video spatial FLOPs (matching window only) for baseline ViViT vs TBKV.

For each video:
  - Baseline: runs the matching-window frames (video[n_cache_frames:]) with
    full attention — counts those FLOPs.
  - TBKV: runs a caching pass on video[:n_cache_frames] to build the KV
    cache, then a matching pass on video[n_cache_frames:] with KV reuse —
    counts only the matching-pass FLOPs.

The y-axis therefore shows the FLOPs actually spent on the matching window,
making the KV-reuse saving directly visible.

Usage:
    python scripts/evaluate/plot_per_frame_flops.py [--n_videos N]
                                                    [--n_cache_frames P]
                                                    [--out PATH]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.5,
    "lines.markersize":   3.5,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.kinetics400 import Kinetics400
from models.vivit import FactorizedViViT
from models.tbkv_vivit import TBKVFactorizedViViT
from utils.config import load_config

CONFIG_DIR  = Path("configs", "evaluate", "vivit_kinetics400")
DEFAULT_OUT = Path("results", "evaluate", "vivit_kinetics400", "per_video_flops.png")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_model(cfg_path, model_class, device):
    cfg_obj = load_config(cfg_path, to_container=False)
    OmegaConf.update(cfg_obj, "_name", Path(cfg_path).stem, merge=True)
    config  = OmegaConf.to_container(cfg_obj, resolve=True)
    model_cfg = dict(config["model"])
    for key in ("vanilla",):
        model_cfg.pop(key, None)
    model = model_class(**model_cfg)
    sd = torch.load(config["weights"], map_location="cpu")
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _sum_counts(module):
    c = module.total_counts()
    return float(sum(c.values())) if c else 0.0


def _matching_flops_baseline(model, video_cache, video_match, device):
    """FLOPs for the matching window under full attention (no KV reuse)."""
    model.clear_counts()
    model.counting()
    with torch.inference_mode():
        # Run the full video so the model sees both segments normally;
        # count only the matching window.
        model.clear_counts()
        _ = model(video_match.to(device))
    flops = _sum_counts(model)
    model.no_counting()
    return flops


def _matching_flops_tbkv(model, video_cache, video_match, device):
    """FLOPs for the matching window under TBKV KV-reuse."""
    model.reset()
    # Caching pass — build KV cache, do not count these FLOPs
    model.set_mode("caching")
    model.no_counting()
    with torch.inference_mode():
        _ = model(video_cache.to(device))

    # Matching pass — count only these FLOPs
    model.set_mode("matching")
    model.clear_counts()
    model.counting()
    with torch.inference_mode():
        _ = model(video_match.to(device))
    flops = _sum_counts(model)
    model.no_counting()
    return flops


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def make_plot(n_videos, n_cache_frames, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    data   = Kinetics400(
        Path("data", "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    loader = DataLoader(data, batch_size=1)

    print("Loading baseline model …")
    model_base = _load_model(CONFIG_DIR / "base.yml", FactorizedViViT, device)
    print("Loading TBKV model …")
    model_tbkv = _load_model(CONFIG_DIR / "tbkv.yml", TBKVFactorizedViViT, device)

    base_gflops_list = []
    tbkv_gflops_list = []

    for idx, (video, _label) in tqdm(
        enumerate(loader), total=n_videos, desc="Videos", ncols=80
    ):
        if idx >= n_videos:
            break
        if video.shape[1] <= n_cache_frames:
            continue                         # skip videos that are too short

        video_cache = video[:, :n_cache_frames]
        video_match = video[:, n_cache_frames:]

        torch.cuda.empty_cache()
        base_gflops = _matching_flops_baseline(
            model_base, video_cache, video_match, device
        ) / 1e9
        torch.cuda.empty_cache()
        tbkv_gflops = _matching_flops_tbkv(
            model_tbkv, video_cache, video_match, device
        ) / 1e9

        base_gflops_list.append(base_gflops)
        tbkv_gflops_list.append(tbkv_gflops)

    xs = list(range(1, len(base_gflops_list) + 1))

    # ── plot ─────────────────────────────────────────────────────────────────
    COLOR_BASE = "0.5"
    COLOR_TBKV = "#1F77B4"

    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    ax.plot(xs, base_gflops_list, "-",  color=COLOR_BASE, linewidth=1.2,
            label="Baseline", zorder=2)
    ax.plot(xs, tbkv_gflops_list, "o-", color=COLOR_TBKV, linewidth=1.5,
            markersize=2.5, label="TBKV", zorder=3)

    ax.set_xlabel("Video index")
    ax.set_ylabel("Matching-window FLOPs (GFLOPs)")
    ax.set_xlim(0.5, len(xs) + 0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="0.8", borderpad=0.5)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_path}  +  {out_path.with_suffix('.pdf')}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_videos",       type=int, default=50,
                        help="Number of validation videos to process")
    parser.add_argument("--n_cache_frames", type=int, default=16,
                        help="Number of caching frames (P)")
    parser.add_argument("--out",            type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()
    make_plot(
        n_videos=args.n_videos,
        n_cache_frames=args.n_cache_frames,
        out_path=args.out,
    )
