#!/usr/bin/env python3
"""
Plot accuracy vs FLOPs from a sweep JSON produced by sweep_tbkv.py.
Parametrised by merge ratio; raw cache shown as a reference star.

Usage:
    python scripts/evaluate/plot_from_sweep.py [--json PATH] [--out PATH]
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
DEFAULT_OUT  = Path("results", "evaluate", "vivit_kinetics400", "flops_accuracy.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(json_path):
    with open(json_path) as f:
        return json.load(f)


def _tbkv_series(data):
    pts = [v for v in data.values() if v["config"] == "tbkv"]
    return sorted(pts, key=lambda p: p["merge_ratio"])


def _raw_points(data):
    """Return all tbkv_raw entries (may differ by merge_ratio)."""
    return [v for v in data.values() if v["config"] == "tbkv_raw"]


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def make_plots(json_path, out_path):
    data = _load(json_path)
    tbkv = _tbkv_series(data)
    raws = _raw_points(data)

    COLOR_T = "#1F77B4"   # muted blue
    COLOR_R = "#D62728"   # muted red

    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    # ── Merged-cache curve (parametrised by r) ──────────────────────────────
    xs = [p["total_flops_per_video"] / 1e9 for p in tbkv]
    ys = [p["top_1"] * 100                  for p in tbkv]
    line, = ax.plot(xs, ys, "o-", color=COLOR_T, zorder=3, label="TBKV (merged cache)")

    # Draw the curve first so transforms are initialised, then annotate
    # perpendicular to the local slope at each point (in display space).
    fig.canvas.draw()
    trans = ax.transData      # data → display pixels
    inv   = ax.transData.inverted()

    def _perp_offset_pts(i, xs, ys, side, dist=9):
        """Return (dx, dy) offset in points perpendicular to the line at index i."""
        import numpy as np
        # Central-difference tangent in display space
        x0 = xs[max(i-1, 0)];        y0 = ys[max(i-1, 0)]
        x1 = xs[min(i+1, len(xs)-1)]; y1 = ys[min(i+1, len(ys)-1)]
        p0 = trans.transform((x0, y0))
        p1 = trans.transform((x1, y1))
        tx, ty = p1[0]-p0[0], p1[1]-p0[1]
        length = (tx**2 + ty**2)**0.5 or 1.0
        # Perpendicular (rotate 90°), then choose side (+1 above, -1 below)
        px, py = -ty/length * side, tx/length * side
        return px * dist, py * dist

    for i, p in enumerate(tbkv):
        if p["merge_ratio"] in (0.3, 0.4):
            continue
        side = 1 if i % 2 == 0 else -1
        dx, dy = _perp_offset_pts(i, xs, ys, side)
        ax.annotate(f"$r\\!=\\!{p['merge_ratio']}$",
                    xy=(p["total_flops_per_video"] / 1e9, p["top_1"] * 100),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=COLOR_T, ha="center",
                    va="bottom" if dy >= 0 else "top")

    # ── Raw-cache reference ──────────────────────────────────────────────────
    for raw in raws:
        xr = raw["total_flops_per_video"] / 1e9
        yr = raw["top_1"] * 100
        ax.scatter([xr], [yr], marker="*", s=70, color=COLOR_R,
                   zorder=4, label="TBKV (raw cache)")
        ax.annotate("raw",
                    xy=(xr, yr),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=6.5, color=COLOR_R, va="bottom", ha="center")

    # ── Axes & style ────────────────────────────────────────────────────────
    all_xs = xs + [r["total_flops_per_video"] / 1e9 for r in raws]
    all_ys = ys + [r["top_1"] * 100                  for r in raws]
    xr_ = max(all_xs) - min(all_xs) or 1.0
    yr_ = max(all_ys) - min(all_ys) or 1.0
    ax.set_xlim(min(all_xs) - 0.08 * xr_, max(all_xs) + 0.30 * xr_)
    ax.set_ylim(min(all_ys) - 0.30 * yr_,  max(all_ys) + 0.20 * yr_)

    ax.set_xlabel("GFLOPs per video (matching pass)")
    ax.set_ylabel("Top-1 Accuracy (\\%)")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(),
              loc="lower left", frameon=True, framealpha=0.95,
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
    make_plots(json_path=args.json, out_path=args.out)
