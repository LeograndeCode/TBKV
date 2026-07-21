#!/usr/bin/env python3
"""
Box-and-whisker template for a TempoMem ablation grid (cache_reuse x
merge_iterations): one box per cache_reuse value, showing the spread of
Top-1 (or mAP) across the merge_iterations settings measured at that
cache_reuse -- a real five-number summary (min, Q1, median, Q3, max) once
populated with actual grid results, n=4 points per box (one per
merge_iterations in {2, 4, 6, 8}).

*** THE DATA BELOW IS SYNTHETIC PLACEHOLDER DATA -- NOT REAL RESULTS. ***
This repo does not currently save per-item outcomes anywhere (every eval
script writes one aggregate number per run, which is why the other paper
figures use Wilson confidence intervals instead of real error bars). A
genuine box needs several distinct real measurements per group; the
cache_reuse x merge_iterations ablation grid (paper_queue5.sh /
paper_vivit_ablation.sh) is the one place in this project that will produce
that. Once it lands, replace SYNTHETIC_DATA below by loading the grid's
per-combo results (see load_real_data() stub) and delete the "SYNTHETIC"
title watermark.

Usage: python scripts/plots/box_whisker_ablation.py [outdir]
Output: <outdir>/fig_box_whisker_ablation.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/comparison/figures")

# ---------------------------------------------------------------- palette ----
# Same categorical palette as tempomem_figures.py, for visual consistency
# across paper figures.
C_TEMPOMEM = "#2a78d6"   # blue - box fill/edge
C_MEDIAN = "#c1272d"     # red  - median line (conventional box-plot accent,
                         # matches the five-number-summary reference figure)
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
MERGE_ITERATIONS = [2, 4, 6, 8]
CACHE_REUSE_VALUES = [0.25, 0.5, 0.75, 0.95]

# SYNTHETIC -- fabricated for layout purposes only. Do not cite these
# numbers anywhere. Shaped as {cache_reuse: [top1 @ mi=2, mi=4, mi=6, mi=8]}.
SYNTHETIC_DATA = {
    0.25: [68.9, 69.4, 68.1, 67.6],
    0.50: [63.2, 64.0, 62.5, 61.8],
    0.75: [57.1, 58.3, 56.4, 55.0],
    0.95: [49.6, 50.8, 48.2, 46.9],
}


def load_real_data():
    """
    Stub for wiring up the real ablation grid once it has finished.
    Expected result dirs (VID example; ViViT is analogous):
      results/evaluate/vitdet_vid/tbkv_eventful_filter_672-token_top_k=[512]-
        warmup=4-cache_reuse=<cr>-merge_iterations=<mi>-n_items=64-
        replay_matching=true/metrics.csv   (column: map_50)
    Parse each (cr, mi) combo's metric into the same
    {cache_reuse: [values across merge_iterations]} shape as SYNTHETIC_DATA
    and return it in place of SYNTHETIC_DATA.
    """
    raise NotImplementedError(
        "Point this at real results once the ablation grid lands; "
        "see the docstring above for the expected directory layout."
    )


def style_axis(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_box_whisker(data, synthetic=True):
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
    style_axis(ax)

    labels = [f"{cr:g}" for cr in CACHE_REUSE_VALUES]
    values = [data[cr] for cr in CACHE_REUSE_VALUES]

    bp = ax.boxplot(
        values,
        labels=labels,
        widths=0.5,
        patch_artist=True,
        whis=1.5,                 # standard Tukey whisker (1.5 x IQR)
        boxprops=dict(facecolor="white", edgecolor=C_TEMPOMEM, linewidth=1.6),
        medianprops=dict(color=C_MEDIAN, linewidth=2.0),
        whiskerprops=dict(color=C_TEMPOMEM, linewidth=1.2, linestyle=(0, (4, 3))),
        capprops=dict(color=C_TEMPOMEM, linewidth=1.2),
        flierprops=dict(
            marker="o", markersize=4.5, markerfacecolor="none",
            markeredgecolor=C_TEMPOMEM, linewidth=0.9,
        ),
        zorder=3,
    )

    ax.set_xlabel("cache_reuse ($\\rho$)")
    ax.set_ylabel("Top-1 accuracy (%)")
    title = ("TempoMem ablation: accuracy spread across\n"
             "merge_iterations $\\in \\{2,4,6,8\\}$, per cache_reuse")
    if synthetic:
        title = "SYNTHETIC PLACEHOLDER -- not real data\n" + title
    ax.set_title(title, fontsize=9, color=(C_MEDIAN if synthetic else INK), pad=8)

    fig.tight_layout()
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        data, synthetic = load_real_data(), False
    except NotImplementedError:
        data, synthetic = SYNTHETIC_DATA, True

    fig = fig_box_whisker(data, synthetic=synthetic)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_box_whisker_ablation.{ext}", bbox_inches="tight")
    tag = "SYNTHETIC placeholder" if synthetic else "real"
    print(f"[{tag}] figure written to {OUT}/")


if __name__ == "__main__":
    main()
