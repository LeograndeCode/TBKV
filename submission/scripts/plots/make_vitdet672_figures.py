#!/usr/bin/env python3
"""
Generate the two ViTDet-B / ImageNet VID (672x672) figures for the paper:

  fig_vid672_frontier.pdf  -- accuracy vs compute, layered PSM vs SOTA
  fig_vid672_ablation.pdf  -- cost-saving / mAP-change ablation, shared zero line

All numbers are measured on the full 639-video validation split unless noted.
Sources:
  dense               results/evaluate/vitdet_vid/base_672/
  Eventful (k=512)    results/evaluate/vitdet_vid/temporal_672/
  STGT (k=512)        results/evaluate/vitdet_vid/stgt_672-token_top_k=[512]/
  MaskVD              maskvd_672 run output (see scripts/run/01_vid_672.sh)
  PSM (layered)       .../psm_eventful_filter_672-token_top_k=[512]-
                      cache_reuse=0.25-merge_iterations=6-warmup=4-
                      replay_matching=true-measure_latency=true/
  ablation grid       .../...-n_items=64-replay_matching=true  (10% split)

Palette: data-viz reference categorical slots 1/2/3/7, validated all-pairs in
light mode (worst CVD dE 9.2, worst normal-vision dE 16.3). Every mark is
direct-labelled, satisfying the relief rule for the sub-3:1 aqua slot.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "paper" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # embed TrueType, no Type 3 (AAAI requirement)
    "ps.fonttype": 42,
})

# ── Figure 1: accuracy vs compute, one point per method ──────────────────────
#            (GFLOPs/frame, mAP@50), colour, marker, label, dx, dy, ha
# Eventful and its spatio-temporal variant share the orange hue (same method
# family); the marker fill distinguishes them, so colour still follows entity.
METHODS = [
    ((174.5, 82.28), INK,    "*", True,  "Dense",              -10,  -3, "right"),
    ((60.7,  81.80), ORANGE, "^", True,  "Eventful",             0,  11, "center"),
    ((52.8,  79.48), ORANGE, "^", False, "Eventful\n(spatio-temporal)",  6, -28, "center"),
    ((68.5,  80.45), AQUA,   "s", True,  "STGT",                11,  -3, "left"),
    ((80.9,  82.05), VIOLET, "D", True,  "MaskVD",              10,   0, "left"),
    ((46.6,  79.87), BLUE,   "o", True,  "PSM\n(layered)", -6,  8, "right"),
]

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

for (x, y), c, m, filled, lab, dx, dy, ha in METHODS:
    ax.plot(x, y, m, color=c if filled else "white", markersize=13 if m == "*" else 9,
            zorder=4, markeredgecolor=c if not filled else "white",
            markeredgewidth=2.0 if not filled else 0.9)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=8, color=INK, linespacing=1.05)

ax.set_xlabel("GFLOPs / frame")
ax.set_ylabel("mAP@50 (%)")
ax.set_xlim(0, 215)
ax.set_ylim(76.0, 84.0)
fig.savefig(OUT / "fig_vid672_frontier.pdf")
fig.savefig(OUT / "fig_vid672_frontier.png")
print("wrote", OUT / "fig_vid672_frontier.pdf")

# ── Figure 2: ablation, single shared zero line ──────────────────────────────
DENSE_GF, DENSE_MAP = 174.5, 82.28
ABL = [  # gamma, best T, GFLOPs/frame, mAP@50   (k=512, 10% split)
    (0.25, 6, 46.6, 79.49),
    (0.50, 6, 32.9, 75.02),
    (0.75, 8, 19.3, 63.63),
    (0.95, 2,  8.4, 38.26),
]
gammas = [g for g, _, _, _ in ABL]
saving = [100 * (1 - gf / DENSE_GF) for _, _, gf, _ in ABL]
dmap = [m - DENSE_MAP for _, _, _, m in ABL]
skip = [100 * (1 - 512 * (1 - g) / 1764) for g in gammas]

# One axes, one zero line. Above the line the unit is % cost saving (full
# scale UP_MAX); below it the unit is mAP@50 points (full scale DN_MAX).
UP_MAX, DN_MAX = 100.0, 50.0
up = [v / UP_MAX for v in saving]
dn = [v / DN_MAX for v in dmap]
x = list(range(len(ABL)))

fig, ax = plt.subplots(figsize=(3.4, 3.1))
ax.bar(x, up, width=0.6, color=BLUE, zorder=3, label="cost saving")
ax.bar(x, dn, width=0.6, color=AQUA, zorder=3, label="mAP@50 change")

for xi, v, n in zip(x, saving, up):
    ax.annotate(f"{v:.1f}", (xi, n), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7.5, color=INK)
for xi, v, n in zip(x, dmap, dn):
    ax.annotate(f"{v:.1f}", (xi, n), textcoords="offset points", xytext=(0, -10),
                ha="center", fontsize=7.5, color=INK)

# ticks: upper half in % saving, lower half in mAP points
yt = [0.0, 0.5, 1.0, -0.2, -0.4, -0.6, -0.8]
yl = ["0", "50", "100", "-10", "-20", "-30", "-40"]
ax.set_yticks(yt)
ax.set_yticklabels(yl)
ax.set_ylim(-1.06, 1.14)
ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.axhline(0, color=INK, linewidth=1.4, zorder=5)   # the single shared zero

ax.text(-0.175, 0.80, "cost saving (%)", transform=ax.transAxes, rotation=90,
        va="center", ha="center", fontsize=9, color=INK)
ax.text(-0.175, 0.19, "mAP@50 change (pts)", transform=ax.transAxes, rotation=90,
        va="center", ha="center", fontsize=9, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([f"{g:g}\n({s:.1f}\\%)".replace("\\", "") for g, s in zip(gammas, skip)])
ax.set_xlabel(r"reuse fraction $\gamma$  (token skipping ratio)")
fig.savefig(OUT / "fig_vid672_ablation.pdf")
fig.savefig(OUT / "fig_vid672_ablation.png")
print("wrote", OUT / "fig_vid672_ablation.pdf")
