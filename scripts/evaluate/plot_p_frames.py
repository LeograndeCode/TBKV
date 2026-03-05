#!/usr/bin/env python3
"""
Plot Top-1 accuracy (y) vs total FLOPs (x) for TBKV merged and raw cache,
with each curve parametrised by the number of caching frames P.

Usage:
    python scripts/evaluate/plot_p_frames.py [--json PATH] [--out PATH]
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    "lines.markersize":   5,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "pdf.fonttype":       42,   # embeds fonts for paper submission
    "ps.fonttype":        42,
})

DEFAULT_JSON = Path("results", "evaluate", "vivit_kinetics400", "sweep_p_frames.json")
DEFAULT_OUT  = Path("results", "evaluate", "vivit_kinetics400", "p_frames.png")
# Optional: point to the base statistics.json to draw a vanilla baseline
DEFAULT_BASE = Path("results", "evaluate", "vivit_kinetics400", "base", "statistics.json")


def _load(json_path):
    with open(json_path) as f:
        return json.load(f)


def _series(data, cfg_name):
    pts = [v for v in data.values() if v["config"] == cfg_name]
    return sorted(pts, key=lambda p: p["n_cache_frames"])


def make_plot(json_path, out_path, base_json=None):
    data = _load(json_path)
    tbkv = _series(data, "tbkv")
    raw  = _series(data, "tbkv_raw")

    # Keep only P values present in BOTH series
    raw_by_p = {p["n_cache_frames"]: p for p in raw}
    tbkv = [p for p in tbkv if p["n_cache_frames"] in raw_by_p]
    raw  = [raw_by_p[p["n_cache_frames"]] for p in tbkv]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    COLOR_T  = "#1F77B4"   # muted blue  — TBKV merged
    COLOR_R  = "#D62728"   # muted red   — TBKV raw
    COLOR_BL = "#2CA02C"   # green       — vanilla baseline

    xs_t = [p["total_flops_per_video"] / 1e9 for p in tbkv]
    ys_t = [p["top_1"] * 100                  for p in tbkv]
    xs_r = [p["total_flops_per_video"] / 1e9 for p in raw]
    ys_r = [p["top_1"] * 100                  for p in raw]

    # ── 1. Connectors between same-P points ─────────────────────────────────
    for t, r in zip(tbkv, raw):
        xt = t["total_flops_per_video"] / 1e9
        xr = r["total_flops_per_video"] / 1e9
        yt = t["top_1"] * 100
        yr = r["top_1"] * 100
        ax.plot([xt, xr], [yt, yr], color="0.75", linewidth=0.7,
                linestyle="-", zorder=1)
        # Label at midpoint
        mx, my = (xt + xr) / 2, (yt + yr) / 2
        P_val = t['n_cache_frames']
        va    = "top" if P_val == 28 else "bottom"
        yoff  = -0.5  if P_val == 28 else  0.0
        ax.text(mx, my + yoff, f"$P\\!={P_val}$",
                ha="center", va=va, fontsize=5.5, color="0.45",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    # ── 2. Main curves ───────────────────────────────────────────────────────
    ax.plot(xs_t, ys_t, "o-", color=COLOR_T, zorder=3, label="TBKV (merged cache)")
    ax.plot(xs_r, ys_r, "s--", color=COLOR_R, zorder=3, label="TBKV (raw cache)")

    # ── 3. Optional vanilla baseline ────────────────────────────────────────
    if base_json and Path(base_json).exists():
        bd = _load(base_json)
        flops_total = bd.get("flops_per_video", {}).get("total", {})
        # sum all flop types
        bl_flops = sum(flops_total.values()) / 1e9 if flops_total else None
        # top-1 from a sibling output.txt is not in statistics.json,
        # so allow passing it via a key or skip if unavailable
        bl_acc = bd.get("top_1_accuracy")
        if bl_flops and bl_acc:
            ax.scatter([bl_flops], [bl_acc * 100], marker="*", s=80,
                       color=COLOR_BL, zorder=4, label="ViViT baseline")
            ax.axvline(bl_flops, color=COLOR_BL, linewidth=0.6,
                       linestyle=":", alpha=0.5)

    # ── 4. Axes & style ─────────────────────────────────────────────────────
    all_xs = xs_t + xs_r
    all_ys = ys_t + ys_r
    xr_ = max(all_xs) - min(all_xs) or 1.0
    yr_ = max(all_ys) - min(all_ys) or 1.0
    ax.set_xlim(min(all_xs) - 0.08 * xr_, max(all_xs) + 0.12 * xr_)
    ax.set_ylim(min(all_ys) - 0.30 * yr_,  max(all_ys) + 0.20 * yr_)

    ax.set_xlabel("GFLOPs per video (matching pass)")
    ax.set_ylabel("Top-1 Accuracy (\\%)")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
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
    parser.add_argument("--json", type=str, default=str(DEFAULT_JSON))
    parser.add_argument("--out",  type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--base", type=str, default=str(DEFAULT_BASE),
                        help="Path to base model statistics.json (optional)")
    args = parser.parse_args()
    make_plot(json_path=args.json, out_path=args.out, base_json=args.base)
