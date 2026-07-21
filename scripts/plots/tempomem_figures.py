#!/usr/bin/env python3
"""Paper figures: TempoMem (ne Eventful-TBKV) vs Eventful vs base (+ MaskVD).

Figure 1  ViViT-B / Kinetics-400   : Top-1 accuracy vs GFLOPs/clip
Figure 2  ViTDet-B / ImageNet-VID  : mAP@50 vs GFLOPs/frame

All numbers are measured in this repo (see results/evaluate/...). TempoMem with
cache_reuse=0 is bit-for-bit the Eventful block, so the Eventful point anchors
the TempoMem curve by construction.

Usage: python scripts/plots/tempomem_figures.py [outdir]
Writes <outdir>/fig_vivit_k400.{pdf,png} and <outdir>/fig_vitdet_vid.{pdf,png}.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/comparison/figures")

# ---------------------------------------------------------------- palette ----
# Validated categorical palette (dataviz reference instance, light mode).
C_TEMPOMEM = "#2a78d6"   # blue   - slot 1 (hero)
C_EVENTFUL = "#008300"   # green  - slot 2
C_MASKVD   = "#e87ba4"   # magenta- slot 3
C_BASE     = "#52514e"   # neutral ink - reference anchor, not a competing hue
GRID       = "#e6e5e1"
INK        = "#0b0b0b"
INK2       = "#52514e"

mpl.rcParams.update({
    "font.size": 9.5,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,   # embed TrueType (camera-ready requirement)
    "ps.fonttype": 42,
})

# ------------------------------------------------------------------- data ----
# ViViT-B factorised / Kinetics-400 FULL val (19,877 videos), two-pass protocol.
# GFLOPs = MATCHING pass only (the caching pass is identical for every method,
# so it is excluded; base has no caching pass and is shown as a reference line).
# Runs: results/evaluate/vivit_kinetics400/{eventful_tbkv_24,eventful_tbkv_48,
# eventful_tbkv_96, eventful_tbkv}; r=96 measured with the final_96
# checkpoint. eventful_tbkv_24/48 ran in this repo; eventful_tbkv_96 and
# eventful_tbkv_48_cr05 (TempoMem on top of r=48) were measured on a
# different server -- see the provenance note in each result dir's
# output.txt.
VIVIT_N = 19877
VIVIT = {
    "base":     {"gflops": 3359, "top1": 73.0},   # vanilla, whole clip
    # Eventful across its own compute knob r = token_top_k (cache_reuse=0).
    # Each r uses its own fine-tuned checkpoint (final_24/48/96).
    "eventful": [   # (r, matching GFLOPs/clip, Top-1 %)
        (24, 435, 62.38),   # Top-5 82.23, caching 618  + matching 435  = 1053
        (48, 860, 67.54),   # Top-5 87.26, caching 1016 + matching 860  = 1877
        (96, 1711, 75.71),  # Top-5 92.35, caching 1814 + matching 1711 = 3525
    ],
    # TempoMem at fixed cache_reuse=rho=0.5, across r (paralleling the
    # Eventful r-sweep above). No r=96 point yet -- no rho=0.5-on-r=96 run.
    "tempomem": [   # (r, matching GFLOPs/clip, Top-1 %)
        (24, 222, 59.85),  # Top-5 79.83, caching 618  + matching 222 = 840
        (48, 435, 58.37),  # Top-5 78.62, caching 1016 + matching 435 = 1451
    ],
}


def wilson_ci(p_pct, n, z=1.96):
    """95% Wilson score interval, returned as (lo, hi) in percent."""
    p = p_pct / 100.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return 100 * (center - half), 100 * (center + half)


def _errbars(pts_pct, n):
    los, his = zip(*(wilson_ci(p, n) for p in pts_pct))
    lo_err = [p - lo for p, lo in zip(pts_pct, los)]
    hi_err = [hi - p for p, hi in zip(pts_pct, his)]
    return [lo_err, hi_err]

# ViTDet-B / ImageNet-VID val @672, 10 videos, frame_stride=1.
# base: all frames. Eventful/TempoMem: 4-frame warm-up excluded (steady state).
# MaskVD: keyframe refresh every `period` frames, all frames scored.
# FILLED_BY_SWEEPS = True  -> replace the placeholders below with sweep output.
VITDET = {
    "base":     {"gflops": 174.5, "map50": 84.1},
    "eventful": [],   # (token_top_k, GF/frame, mAP50)  <- cr 0.0 runs
    "tempomem": [],   # (token_top_k, GF/frame, mAP50)  <- cr 0.5 runs
    "maskvd":   [],   # (period,      GF/frame, mAP50)
}


def style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_vivit():
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
    style_axis(ax)

    # Base accuracy as a reference line (its 3359 GF/clip whole-clip cost is
    # off this axis; keeping it as a line preserves the origin-anchored view).
    b = VIVIT["base"]
    ax.axhline(b["top1"], color=C_BASE, linewidth=1, linestyle=(0, (4, 3)),
               zorder=2)
    ax.annotate(f"ViViT-B base: {b['top1']:.0f}%  ({b['gflops']} GF/clip, 100 clips)",
                (0.02, b["top1"]), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 4), fontsize=7.5,
                color=INK2)

    # Eventful across its own knob r.
    ev = sorted((p for p in VIVIT["eventful"] if p[1] is not None),
                key=lambda p: p[1])
    ev_x = [g for _, g, _ in ev]
    ev_y = [t for _, _, t in ev]
    ax.errorbar(ev_x, ev_y, yerr=_errbars(ev_y, VIVIT_N), fmt="-s",
                color=C_EVENTFUL, linewidth=1.8, markersize=5.5, capsize=2.5,
                elinewidth=0.9, zorder=4, label="Eventful (varying r)")
    ev_off = {3: (-7, 8, "right"), 6: (7, 8, "left"), 12: (7, 7, "left"),
              24: (7, 7, "left"), 48: (0, -17, "center")}
    for r, g, t in ev:
        dx, dy, ha = ev_off.get(r, (7, 7, "left"))
        ax.annotate(f"r={r}", (g, t), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7.5, color=C_EVENTFUL)

    # TempoMem at fixed rho=0.5, across r.
    tm = sorted(VIVIT["tempomem"], key=lambda p: p[1])
    tm_x = [g for _, g, _ in tm]
    tm_y = [t for _, _, t in tm]
    ax.errorbar(tm_x, tm_y, yerr=_errbars(tm_y, VIVIT_N), fmt="-o",
                color=C_TEMPOMEM, linewidth=1.8, markersize=5.5, capsize=2.5,
                elinewidth=0.9, zorder=5,
                label="Eventful + TempoMem ($\\rho$=0.5, varying r)")
    tm_off = {24: (0, -15, "center"), 48: (0, -15, "center")}
    for r, g, t in tm:
        dx, dy, ha = tm_off.get(r, (0, -15, "center"))
        ax.annotate(f"r={r}", (g, t), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7.5, color=C_TEMPOMEM)

    ax.set_xlim(0, None)
    # Zoomed to the data band (not 0-80) so the Wilson CI whiskers -- a few
    # tenths of a point wide at n=19,877 -- are actually visible.
    ax.set_ylim(55, 78)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.set_xlabel("GFLOPs / clip")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Effect of TempoMem on the Eventful Transformer\n"
                 "ViViT-B · Kinetics-400 full val (19,877 clips) · Wilson 95% CI",
                 fontsize=9, color=INK, pad=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_vivit_k400.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_vitdet():
    if not (VITDET["eventful"] and VITDET["tempomem"]):
        print("ViTDet data not filled in yet - skipping figure 2")
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.1), dpi=200)
    style_axis(ax)

    tm = sorted(VITDET["tempomem"], key=lambda r: r[1])
    ev = sorted(VITDET["eventful"], key=lambda r: r[1])
    ax.plot([g for _, g, _ in tm], [m for _, _, m in tm], "-o",
            color=C_TEMPOMEM, linewidth=2, markersize=5.5, zorder=5,
            label="TempoMem (ours)", clip_on=False)
    ax.plot([g for _, g, _ in ev], [m for _, _, m in ev], "-s",
            color=C_EVENTFUL, linewidth=2, markersize=5.5, zorder=4,
            label="Eventful", clip_on=False)
    for k, g, m in tm:
        ax.annotate(f"r={k}", (g, m), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7.5, color=INK2)

    if VITDET["maskvd"]:
        mv = sorted(VITDET["maskvd"], key=lambda r: r[1])
        ax.plot([g for _, g, _ in mv], [m for _, _, m in mv], "-D",
                color=C_MASKVD, linewidth=2, markersize=5.5, zorder=3,
                label="MaskVD", clip_on=False)
        for p, g, m in mv:
            ax.annotate(f"T={p}", (g, m), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=INK2)

    b = VITDET["base"]
    ax.plot(b["gflops"], b["map50"], "*", color=C_BASE, markersize=12,
            zorder=5, label="ViTDet-B (base)", clip_on=False)

    ax.set_xlabel("GFLOPs / frame (backbone)")
    ax.set_ylabel("mAP@50 (%)")
    ax.set_title("ImageNet-VID · ViTDet-B @672 (10 videos)",
                 fontsize=9.5, color=INK, pad=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_vitdet_vid.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_vivit()
    fig_vitdet()
    print(f"figures written to {OUT}/")
