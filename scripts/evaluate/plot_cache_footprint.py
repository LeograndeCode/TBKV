#!/usr/bin/env python3
"""
Plot cache footprint (MB) vs total memory operations (MB) for TBKV variants.

  y-axis: avg K+V cache size in MB (all blocks summed, per video)
  x-axis: avg total memory ops in MB = cache writes + cache reads

Usage:
    python scripts/evaluate/plot_cache_footprint.py [--json PATH] [--out PATH]
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
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

DEFAULT_JSON = Path("results", "evaluate", "vivit_kinetics400", "sweep_statistics.json")
DEFAULT_OUT  = Path("results", "evaluate", "vivit_kinetics400", "cache_footprint.png")


def _load(json_path):
    with open(json_path) as f:
        return json.load(f)


def _tbkv_series(data):
    pts = [v for v in data.values() if v["config"] == "tbkv"]
    return sorted(pts, key=lambda p: p["merge_ratio"])


def _raw_points(data):
    return [v for v in data.values() if v["config"] == "tbkv_raw"]


def _ensure_memory_fields(p):
    if "avg_total_memory_ops_bytes" not in p:
        write_b = p.get("avg_cache_kv_mb", 0.0) * 1e6
        read_b  = p.get("avg_cache_kv_total_read_bytes",
                         p.get("avg_cache_kv_read_bytes_per_matching_frame", 0.0))
        p["avg_total_memory_ops_bytes"] = write_b + read_b
    return p


def make_plot(json_path, out_path):
    data = _load(json_path)
    tbkv = [_ensure_memory_fields(p) for p in _tbkv_series(data)]
    raws = [_ensure_memory_fields(p) for p in _raw_points(data)]

    COLOR_T = "#1F77B4"
    COLOR_R = "#D62728"

    to_mb = 1 / 1e6
    xs = [p["avg_total_memory_ops_bytes"] * to_mb for p in tbkv]
    ys = [p["avg_cache_kv_mb"]                    for p in tbkv]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(xs, ys, "o-", color=COLOR_T, zorder=3, label="TBKV (merged cache)")

    # Perpendicular label placement
    fig.canvas.draw()
    trans = ax.transData

    def _perp_offset(i, xs, ys, side, dist=9):
        x0, y0 = xs[max(i-1, 0)],         ys[max(i-1, 0)]
        x1, y1 = xs[min(i+1, len(xs)-1)], ys[min(i+1, len(ys)-1)]
        p0 = trans.transform((x0, y0))
        p1 = trans.transform((x1, y1))
        tx, ty = p1[0]-p0[0], p1[1]-p0[1]
        length = (tx**2 + ty**2)**0.5 or 1.0
        px, py = -ty/length * side, tx/length * side
        return px * dist, py * dist

    SKIP_R = {0.3, 0.4}
    for i, p in enumerate(tbkv):
        if p["merge_ratio"] in SKIP_R:
            continue
        side = 1 if i % 2 == 0 else -1
        dx, dy = _perp_offset(i, xs, ys, side)
        ax.annotate(f"$r\\!=\\!{p['merge_ratio']}$",
                    xy=(p["avg_total_memory_ops_bytes"] * to_mb, p["avg_cache_kv_mb"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=COLOR_T, ha="center",
                    va="bottom" if dy >= 0 else "top")

    for raw in raws:
        rx = raw["avg_total_memory_ops_bytes"] * to_mb
        ry = raw["avg_cache_kv_mb"]
        ax.scatter([rx], [ry], marker="*", s=70, color=COLOR_R,
                   zorder=4, label="TBKV (raw cache)")
        ax.annotate("raw", xy=(rx, ry),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=6.5, color=COLOR_R, ha="center", va="bottom")

    all_xs = xs + [r["avg_total_memory_ops_bytes"] * to_mb for r in raws]
    all_ys = ys + [r["avg_cache_kv_mb"]                    for r in raws]
    xr_ = max(all_xs) - min(all_xs) or 1.0
    yr_ = max(all_ys) - min(all_ys) or 1.0
    ax.set_xlim(min(all_xs) - 0.08 * xr_, max(all_xs) + 0.25 * xr_)
    ax.set_ylim(min(all_ys) - 0.30 * yr_,  max(all_ys) + 0.20 * yr_)

    ax.set_xlabel("Memory ops per video (writes + reads, MB)")
    ax.set_ylabel("Cache K+V footprint (MB)")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(),
              loc="lower right", frameon=True, framealpha=0.95,
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
    args = parser.parse_args()
    make_plot(json_path=args.json, out_path=args.out)
