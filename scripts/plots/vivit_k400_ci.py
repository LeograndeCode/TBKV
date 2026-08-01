#!/usr/bin/env python3
"""
Top-1 accuracy vs matching GFLOPs for Eventful and Eventful+TempoMem (PSM) on
ViViT-B / Kinetics-400, across the host token budgets k in {24, 48, 96}.
Full 19,877-clip validation split; numbers are the ones in evaluation.tex
tab:k400.

ERROR BARS. Top-1 is a proportion over N=19,877 clips, so it carries binomial
sampling uncertainty and gets a Wilson 95% score interval -- the same estimator
tempomem_figures.py uses, so the two figures agree. At this N the interval is
about +-0.7 points, which is why the y-axis is zoomed to the data band rather
than anchored at zero: at 0-100 the whiskers would be invisible.

The x axis carries NO error bar, deliberately. Matching GFLOPs is an exact
count from in-code operation counters, not an estimate -- it has no sampling
distribution, and drawing one would be a fabrication.

PSM has no k=96 run; that point is simply absent rather than interpolated.

Usage:  python scripts/plots/vivit_k400_ci.py [outdir]
Output: paper/Figures/fig_k400_ci.{png,pdf} + a copy in <outdir>
        (default outdir: results/comparison/figures)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "comparison" / "figures"

# ---------------------------------------------------------------- palette ----
# Same slots as tempomem_figures.py. Validated with the dataviz
# validate_palette.js (light, surface #fcfcfb): lightness band, chroma floor,
# CVD separation, normal-vision floor and contrast all PASS. The tritan pair
# sits in the 6-8 floor band, legal only with secondary encoding -- hence the
# distinct marker shapes (square vs circle) and the per-point k labels.
C_PSM = "#2a78d6"        # blue  - slot 1
C_EVENTFUL = "#008300"   # green - slot 2
C_DENSE = "#52514e"      # neutral ink - reference anchor, not a competing hue
GRID = "#e6e5e1"
INK = "#0b0b0b"
INK2 = "#52514e"

matplotlib.rcParams.update({
    "font.size": 9.5,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------------------------------------------- data ----
# evaluation.tex tab:k400, full validation split, single-pass accounting.
# PSM rows use the ablation-selected gamma=0.95, T=2.
# Provenance (see scripts/plots/vivit_k400_budget_dumbbell.py for the full
# table): the k=96 Eventful and k=48 PSM cells are author-supplied and not
# reproducible from this checkout; the rest parse from local result dirs.
N_CLIPS = 19877
DENSE_TOP1 = 78.64

# (k, matching GFLOPs/clip, Top-1 %)
EVENTFUL = [
    (24, 435, 62.38),
    (48, 860, 67.54),
    (96, 1711, 75.71),
]
PSM = [
    (24, 27, 59.82),
    (48, 45, 58.37),
    # k=96: not run.
]


def wilson_ci(p_pct, n, z=1.96):
    """95% Wilson score interval, returned as (lo, hi) in percent."""
    p = p_pct / 100.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return 100 * (center - half), 100 * (center + half)


def errbars(pts_pct, n):
    los, his = zip(*(wilson_ci(p, n) for p in pts_pct))
    return [[p - lo for p, lo in zip(pts_pct, los)],
            [hi - p for p, hi in zip(pts_pct, his)]]


def style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def build():
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)
    style_axis(ax)

    ax.axhline(DENSE_TOP1, color=C_DENSE, linewidth=1.1, linestyle=(0, (5, 3)),
               zorder=2)
    ax.annotate(f"dense ViViT-B  {DENSE_TOP1:.2f}%", (0.015, DENSE_TOP1),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                xytext=(0, 4), fontsize=7.5, color=C_DENSE)

    series = [
        ("Eventful (host)", EVENTFUL, C_EVENTFUL, "-s",
         {24: (8, -12, "left"), 48: (8, -12, "left"), 96: (-8, -12, "right")}),
        ("$+$ PSM ($\\gamma$=0.95, $T$=2)", PSM, C_PSM, "-o",
         {24: (0, 11, "center"), 48: (8, 6, "left")}),
    ]

    for label, pts, color, fmt, offsets in series:
        pts = sorted(pts, key=lambda p: p[1])
        xs = [g for _, g, _ in pts]
        ys = [t for _, _, t in pts]
        ax.errorbar(xs, ys, yerr=errbars(ys, N_CLIPS), fmt=fmt, color=color,
                    linewidth=1.8, markersize=5.5, capsize=2.5, elinewidth=0.9,
                    zorder=4, label=label)
        for k, g, t in pts:
            dx, dy, ha = offsets.get(k, (8, 6, "left"))
            ax.annotate(f"$k$={k}", (g, t), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=7.5, color=color)

    ax.set_xscale("log")
    ax.set_xlim(18, 3000)
    # Wide enough to breathe, tight enough that the Wilson whiskers (~0.7 pt at
    # N=19,877) stay visible; anchoring at 0 would hide them entirely.
    ax.set_ylim(52, 84)

    # A log axis defaults to decade-only ticks, which left just 10^2 and 10^3
    # labelled. Label real GFLOPs values across the whole range instead.
    xticks = [20, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:,}" for t in xticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("Matching GFLOPs / clip (log scale)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Eventful vs PSM", fontsize=10, color=INK, pad=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Wilson 95% CI on Top-1, N={N_CLIPS:,} clips")
    for name, pts in (("Eventful", EVENTFUL), ("+ PSM", PSM)):
        for k, g, t in pts:
            lo, hi = wilson_ci(t, N_CLIPS)
            print(f"  {name:9s} k={k:<3} {t:6.2f}%  [{lo:.2f}, {hi:.2f}]  "
                  f"(+-{(hi - lo) / 2:.2f})   matching {g:>5} GF/clip")
    print("  + PSM     k=96  not run")

    fig = build()
    # paper/Figures is where \includegraphics resolves from (graphicspath);
    # the results/comparison copy keeps it beside the other comparison output.
    targets = [REPO / "paper" / "Figures", OUT]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(target / f"fig_k400_ci.{ext}", bbox_inches="tight")
        print(f"\nfigure written to {target}/fig_k400_ci.{{png,pdf}}")


if __name__ == "__main__":
    main()
