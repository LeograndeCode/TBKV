#!/usr/bin/env python3
"""
Eventful vs TempoMem (PSM) on ViViT-B / Kinetics-400 across the three host
token budgets k in {24, 48, 96}, full 19,877-clip validation split.

FORM NOTE -- why this is not a box plot.
A box is a five-number summary and needs several distinct measurements per
group. There are none to be had here:
  * Eventful has exactly ONE configuration per k -- gamma does not apply to it
    -- so its spread at every budget is a single point, by construction.
  * TempoMem has at most two full-split runs at any k, and ZERO at k=96.
So the honest form for "host vs host+PSM at each budget" is a dumbbell (the
before->after-per-item form), which is what this script draws. The one place a
genuine box exists in this project is TempoMem's gamma sweep at k=24 on the
10% subsample (n=4) -- that belongs to box_whisker_ablation.py, not here.

Two panels, never a dual axis: Top-1 and matching GFLOPs are different scales
and get their own axis each.

PROVENANCE. Only three of the six cells are reproducible from this checkout.
Points parsed from local result dirs are drawn as FILLED markers; points whose
output.txt states the numbers were supplied by the author (measured off this
checkout) are drawn HOLLOW. The missing k=96 TempoMem cell is drawn as an
explicit gap, never interpolated.

Usage:  python scripts/plots/vivit_k400_budget_dumbbell.py [outdir]
Output: <outdir>/fig_k400_budget_dumbbell.{png,pdf}
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
VIVIT = REPO / "results" / "evaluate" / "vivit_kinetics400"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "comparison" / "figures"

# ---------------------------------------------------------------- palette ----
# Same slots as tempomem_figures.py, so this figure sits beside the others.
# Validated (dataviz validate_palette.js, light, surface #fcfcfb): lightness
# band PASS, chroma PASS, CVD separation PASS (worst adjacent dE 26.5 protan /
# 7.6 tritan), normal-vision PASS (dE 29.0), contrast PASS. The tritan pair
# lands in the 6-8 floor band, which is legal only with secondary encoding --
# hence the direct labels on every marker below.
C_PSM = "#2a78d6"       # blue  - slot 1
C_EVENTFUL = "#008300"  # green - slot 2
GRID = "#e6e5e1"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"

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

BUDGETS = [24, 48, 96]

# Result dir per (method, k). None = no such run exists anywhere.
SOURCES = {
    ("eventful", 24): "eventful_tbkv_24",
    ("eventful", 48): "eventful_tbkv_48",
    ("eventful", 96): "eventful_tbkv_96",
    ("psm", 24): "eventful_tbkv_24-cache_reuse=0.95-merge_iterations=2",
    # gamma=0.95, T=2, non-replay -- the row used in the paper's tab:k400.
    # Its output.txt records author-supplied numbers; see EXTERNAL detection.
    ("psm", 48): "eventful_tbkv_48_cr05",
    ("psm", 96): None,
}

# Overrides for cells whose paper value does not come from the local dir.
# The k=48 PSM point in tab:k400 (58.37 / 1062 whole / 45 matching) is the
# gamma=0.95 non-replay run, which was measured off this checkout; the local
# `eventful_tbkv_48_cr05` dir is the gamma=0.5 run (58.37 / 1451 / 435) and the
# local gamma=0.95 dir is replay-protocol (58.45, no whole-clip term). Neither
# local dir is the paper's cell, so it is stated here explicitly.
OVERRIDES = {
    ("psm", 48): dict(top1=58.37, matching=45.0, whole=1062.0, external=True,
                      note="gamma=0.95 non-replay, author-supplied"),
}


def parse_run(dirname):
    """Read Top-1 and the caching/matching GFLOPs split from a run's output.txt.

    Returns dict(top1, matching, whole, external, note) or None if unparseable.
    `external` is taken from the run's own provenance banner, not guessed.
    """
    out = VIVIT / dirname / "output.txt"
    if not out.exists():
        return None
    text = out.read_text()
    top1 = re.findall(r"Top-1 Accuracy\s*:\s*([\d.]+)%", text)
    if not top1:
        return None
    split = re.findall(
        r"caching\s+([\d.]+)\s*\+\s*matching\s+([\d.]+)\s*=\s*TOTAL\s+([\d.]+)", text
    )
    if split:
        caching, matching, whole = (float(v) for v in split[-1])
    else:
        # replay-protocol output: matching only, caching discarded
        m = re.findall(r"matching\s+([\d.]+)\s*\(caching", text)
        if not m:
            return None
        matching, whole = float(m[-1]), float("nan")
    external = bool(re.search(r"supplied by the author", text))
    return dict(top1=float(top1[-1]), matching=matching, whole=whole,
                external=external, note="author-supplied" if external else "local run")


def load():
    data = {}
    for (method, k), dirname in SOURCES.items():
        if (method, k) in OVERRIDES:
            data[(method, k)] = OVERRIDES[(method, k)]
            continue
        data[(method, k)] = parse_run(dirname) if dirname else None
    return data


def style_axis(ax):
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def marker_kw(color, external):
    """Hollow marker = number not reproducible from this checkout."""
    if external:
        return dict(marker="o", markersize=8.5, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=2.0)
    return dict(marker="o", markersize=8.5, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=2.0)


def draw_panel(ax, data, field, xlabel, log=False, fmt="{:.2f}"):
    style_axis(ax)
    ys = list(range(len(BUDGETS)))[::-1]  # k=24 on top

    for y, k in zip(ys, BUDGETS):
        ev, psm = data[("eventful", k)], data[("psm", k)]
        if ev and psm:
            ax.plot([ev[field], psm[field]], [y, y], color=MUTED,
                    linewidth=1.6, zorder=2, solid_capstyle="round")
        for rec, color, name in ((ev, C_EVENTFUL, "Eventful"), (psm, C_PSM, "PSM")):
            if not rec:
                continue
            ax.plot([rec[field]], [y], linestyle="none", zorder=3,
                    **marker_kw(color, rec["external"]))
        # Direct labels (secondary encoding; required by the tritan pair).
        if ev and psm:
            lo, hi = sorted((ev, psm), key=lambda r: r[field])
            lo_c = C_EVENTFUL if lo is ev else C_PSM
            hi_c = C_EVENTFUL if hi is ev else C_PSM
            ax.annotate(fmt.format(lo[field]), (lo[field], y), textcoords="offset points",
                        xytext=(-9, 0), ha="right", va="center", fontsize=8, color=INK2)
            ax.annotate(fmt.format(hi[field]), (hi[field], y), textcoords="offset points",
                        xytext=(9, 0), ha="left", va="center", fontsize=8, color=INK2)
            del lo_c, hi_c
        elif ev:
            ax.annotate(fmt.format(ev[field]), (ev[field], y), textcoords="offset points",
                        xytext=(9, 0), ha="left", va="center", fontsize=8, color=INK2)
            ax.annotate("PSM not run", (ev[field], y), textcoords="offset points",
                        xytext=(-9, 0), ha="right", va="center", fontsize=8,
                        color=MUTED, style="italic")

    if log:
        ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"$k$ = {k}" for k in BUDGETS])
    ax.set_ylim(-0.6, len(BUDGETS) - 0.4)
    ax.set_xlabel(xlabel)
    # Generous x margin: every marker carries an offset direct label, and the
    # leftmost one otherwise collides with the spine.
    ax.margins(x=0.30)


def build(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=200)

    draw_panel(axes[0], data, "top1", "Top-1 accuracy (%)", fmt="{:.2f}")
    draw_panel(axes[1], data, "matching", "Matching GFLOPs / clip (log)",
               log=True, fmt="{:.0f}")

    axes[0].set_title("Accuracy", fontsize=9.5, color=INK, pad=6, loc="left")
    axes[1].set_title("Steady-state compute", fontsize=9.5, color=INK, pad=6, loc="left")

    handles = [
        Line2D([], [], linestyle="none", label="Eventful (host)",
               **marker_kw(C_EVENTFUL, False)),
        Line2D([], [], linestyle="none", label="$+$ PSM ($\\gamma$=0.95, $T$=2)",
               **marker_kw(C_PSM, False)),
        Line2D([], [], linestyle="none", markerfacecolor="white",
               markeredgecolor=MUTED, marker="o", markersize=8.5,
               markeredgewidth=2.0, label="hollow: author-supplied"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.10), fontsize=8.5, handletextpad=0.4,
               columnspacing=1.6)

    fig.suptitle("ViViT-B / Kinetics-400, full 19,877-clip validation split",
                 fontsize=10, color=INK, y=1.04)
    fig.tight_layout()
    return fig


def main():
    data = load()

    print(f"{'method':>9} {'k':>4} {'Top-1':>7} {'match GF':>9} {'whole GF':>9}  provenance")
    for k in BUDGETS:
        for method in ("eventful", "psm"):
            rec = data[(method, k)]
            if rec is None:
                print(f"{method:>9} {k:>4} {'--':>7} {'--':>9} {'--':>9}  NO RUN EXISTS")
                continue
            whole = "--" if rec["whole"] != rec["whole"] else f"{rec['whole']:.0f}"
            print(f"{method:>9} {k:>4} {rec['top1']:7.2f} {rec['matching']:9.0f} "
                  f"{whole:>9}  {rec['note']}")

    missing = [key for key, rec in data.items() if rec is None]
    external = [key for key, rec in data.items() if rec and rec["external"]]
    print(f"\n{len(missing)} cell(s) with no run: {missing}")
    print(f"{len(external)} cell(s) not reproducible from this checkout: {external}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig = build(data)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_k400_budget_dumbbell.{ext}", bbox_inches="tight")
    print(f"\nfigure written to {OUT}/fig_k400_budget_dumbbell.{{png,pdf}}")


if __name__ == "__main__":
    main()
