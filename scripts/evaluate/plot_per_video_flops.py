#!/usr/bin/env python3
"""
Plot cumulative KV-projection FLOPs over N validation videos:
  Baseline  — full K+V projection for every token on every matching-window frame
  TBKV      — K+V projection only for unmatched tokens (KV reuse saves the rest)

X-axis: video index
Y-axis: cumulative KV-projection GFLOPs  (running total)

Usage:
    python scripts/evaluate/plot_per_video_flops.py [--n_videos N]
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
from eventful_transformer.counting import CountedLinear
from eventful_transformer.tbkv_blocks import TBKVBlock
from utils.config import load_config

CONFIG_DIR  = Path("configs", "evaluate", "vivit_kinetics400")
DEFAULT_OUT = Path("results", "evaluate", "vivit_kinetics400", "cumulative_kv_flops.png")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_model(cfg_path, model_class, device):
    cfg_obj = load_config(cfg_path, to_container=False)
    OmegaConf.update(cfg_obj, "_name", Path(cfg_path).stem, merge=True)
    config = OmegaConf.to_container(cfg_obj, resolve=True)
    model_cfg = dict(config["model"])
    for key in ("vanilla",):
        model_cfg.pop(key, None)
    model = model_class(**model_cfg)
    sd = torch.load(config["weights"], map_location="cpu")
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _kv_flops_from_baseline(model):
    """
    Sum K+V projection FLOPs from baseline model after a counting pass.
    Baseline uses a combined qkv CountedLinear; K and V together are 2/3 of it.
    """
    total = 0.0
    for module in model.modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, CountedLinear):
            c = module.qkv.total_counts()
            total += float(c.get("linear_flops", 0)) * (2.0 / 3.0)
    return total


def _kv_flops_from_tbkv(model):
    """
    Sum K+V projection FLOPs from TBKV model after a counting pass.
    TBKV uses separate self.k and self.v CountedLinear projections,
    and only computes them for tokens that weren't matched to the cache.
    """
    total = 0.0
    for module in model.modules():
        if isinstance(module, TBKVBlock):
            ck = module.k.total_counts()
            cv = module.v.total_counts()
            total += float(ck.get("linear_flops", 0)) + float(cv.get("linear_flops", 0))
    return total


def _run_matching_window(model, video_cache, video_match, device, is_tbkv):
    """
    Run caching pass (build KV cache, no counting) then matching pass
    (with counting) on video_match.  Returns KV-projection FLOPs.
    """
    if is_tbkv:
        model.reset()
        model.set_mode("caching")
        model.no_counting()
        with torch.inference_mode():
            _ = model(video_cache.to(device))
        model.set_mode("matching")
    # else: baseline — just forward on the matching window directly

    model.clear_counts()
    model.counting()
    with torch.inference_mode():
        _ = model(video_match.to(device))
    model.no_counting()

    if is_tbkv:
        return _kv_flops_from_tbkv(model)
    else:
        return _kv_flops_from_baseline(model)


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

    base_cumulative = []
    tbkv_cumulative = []
    cum_base = 0.0
    cum_tbkv = 0.0

    for idx, (video, _label) in tqdm(
        enumerate(loader), total=n_videos, desc="Videos", ncols=80
    ):
        if idx >= n_videos:
            break
        if video.shape[1] <= n_cache_frames:
            continue

        video_cache = video[:, :n_cache_frames]
        video_match = video[:, n_cache_frames:]

        torch.cuda.empty_cache()
        cum_base += _run_matching_window(
            model_base, video_cache, video_match, device, is_tbkv=False
        )
        torch.cuda.empty_cache()
        cum_tbkv += _run_matching_window(
            model_tbkv, video_cache, video_match, device, is_tbkv=True
        )

        base_cumulative.append(cum_base / 1e9)
        tbkv_cumulative.append(cum_tbkv / 1e9)

    xs = list(range(1, len(base_cumulative) + 1))

    # ── plot ─────────────────────────────────────────────────────────────────
    COLOR_BASE = "0.5"
    COLOR_TBKV = "#1F77B4"

    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    ax.plot(xs, base_cumulative, "-",  color=COLOR_BASE, linewidth=1.5,
            label="Baseline KV", zorder=2)
    ax.plot(xs, tbkv_cumulative, "-",  color=COLOR_TBKV, linewidth=1.5,
            label="TBKV KV", zorder=3)

    ax.set_xlabel("Video index")
    ax.set_ylabel("Cumulative KV FLOPs (GFLOPs)")
    ax.set_xlim(0.5, len(xs) + 0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.95,
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
