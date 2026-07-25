#!/usr/bin/env python3
"""
Generate the ViViT / Kinetics-400 accuracy-compute frontier figure:

  fig_k400_frontier.pdf  -- Top-1 vs steady-state GFLOPs/frame,
                             Eventful (r=24/48/96) vs layered TempoMem (r=24/48)

All numbers are measured on the full 19,877-clip validation split unless
noted. Steady-state GFLOPs/frame = matching-pass GFLOPs/clip / 12 (16-frame
clips, 4 caching frames), matching the convention already used for the VID
frontier figure's per-frame axis.

Sources:
  dense                results/evaluate/vivit_kinetics400/base/
  Eventful  (r=24)      results/evaluate/vivit_kinetics400/eventful_tbkv_24/
  Eventful  (r=48)      results/evaluate/vivit_kinetics400/eventful_tbkv_48/
  Eventful  (r=96)*     results/evaluate/vivit_kinetics400/eventful_tbkv_96/
  TempoMem  (r=24,cr=0.5)  results/evaluate/vivit_kinetics400/eventful_tbkv/
  TempoMem  (r=48,cr=0.5)* results/evaluate/vivit_kinetics400/eventful_tbkv_48_cr05/
  TempoMem  (r=96,cr=0.5)  NOT YET RUN -- no result directory exists.

* flagged in their own output.txt as "measured on a different server --
  not run through this repo checkout; numbers supplied by the author" --
  drawn as hollow markers / dashed segments to mark them as unverified
  in-pipeline, per the same convention as the VID frontier's open markers.

Palette: reuses the paper's already-validated categorical pair (BLUE for
TempoMem, ORANGE for Eventful) from make_vitdet672_figures.py, re-checked
here with the dataviz skill's validator restricted to the two true
categorical hues (INK is a neutral reference marker, not a series color):
CVD dE 24.7 (protan), normal-vision floor 33.6 -- both well clear of the
8.0 / 15.0 targets.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "paper" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# ── data: (GFLOPs/frame, Top-1 %) ────────────────────────────────────────────
DENSE = (210.0, 78.64)

EVENTFUL = [  # r, (gf, top1), verified
    (24, (36.25, 62.38), True),
    (48, (71.7, 67.54), True),
    (96, (142.6, 75.71), False),
]
TEMPOMEM = [  # r, (gf, top1), verified
    (24, (18.5, 59.85), True),
    (48, (36.25, 58.37), False),
    # r=96: not yet run -- intentionally absent, called out in the caption.
]

fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)


def plot_curve(points, color, label, label_dy):
    """Connect a method's r-sweep; dash any segment touching an unverified point."""
    for (r0, p0, v0), (r1, p1, v1) in zip(points, points[1:]):
        style = "-" if (v0 and v1) else "--"
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], style, color=color,
                 linewidth=1.4, zorder=2)
    for i, (r, (x, y), verified) in enumerate(points):
        ax.plot(x, y, "o", color=color if verified else "white",
                 markeredgecolor=color, markeredgewidth=1.8 if not verified else 0.9,
                 markersize=9, zorder=4)
        dy = label_dy if i % 2 == 0 else -label_dy - 12
        ax.annotate(f"$r$={r}", (x, y), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=7.5, color=INK)
    ax.plot([], [], "o-", color=color, label=label, markersize=6)


plot_curve(EVENTFUL, ORANGE, "Eventful", 9)
plot_curve(TEMPOMEM, BLUE, "TempoMem (layered, $\\gamma$=0.5)", 9)

ax.plot(*DENSE, "*", color=INK, markersize=13, zorder=4)
ax.annotate("Dense", DENSE, textcoords="offset points", xytext=(-10, -3),
            ha="right", fontsize=8, color=INK)

# Flag the missing TempoMem r=96 point rather than fabricate one.
ax.annotate("TempoMem\n$r$=96: TBD", (142.6, 61.5), fontsize=7, color=INK2,
            ha="center", style="italic")

ax.set_xlabel("Steady-state GFLOPs / frame")
ax.set_ylabel("Top-1 (%)")
ax.set_xlim(0, 225)
ax.set_ylim(55, 82)
ax.legend(loc="lower right", fontsize=7, frameon=False)

fig.savefig(OUT / "fig_k400_frontier.pdf")
fig.savefig(OUT / "fig_k400_frontier.png")
print("wrote", OUT / "fig_k400_frontier.pdf")
