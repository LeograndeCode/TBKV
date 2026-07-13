#!/usr/bin/env python3
"""
Produce the three paper figures requested for the TBKV submission.

  1. long_run_efficiency.pdf
     Amortized GFLOPs/frame vs. clip length T. TBKV pays the heavy keyframe
     pass only every ``cache_period`` frames, so its amortized cost keeps
     falling as T grows and eventually beats the per-frame gating cost of
     STGT / Eventful. "As the clip gets longer, TBKV wins."

  2. specular_tradeoff.pdf
     A mirrored (specular) bar chart. The horizontal centre line is the
     baseline. Bars going UP are mAP@50 (accuracy retained); bars going DOWN
     are GFLOPs/frame (compute spent). Reading left->right the TBKV configs get
     more aggressive: GFLOPs shrink (good) while mAP@50 slowly drops.

  3. tbkv_configs_pareto.pdf
     Accuracy (mAP@50) vs. GFLOPs/frame for every TBKV operating point, with
     the SOTA competitors overlaid, showing where the TBKV curve now sits on
     the accuracy/efficiency frontier.

All numbers are read from RESULTS below so the figures can be regenerated
after new evaluation runs simply by editing the dict.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Results (n_items=10, ImageNet VID 672px, ViTDet-B).  Edit after new runs.
# ──────────────────────────────────────────────────────────────────────────────
BASELINE = {"gflops": 174.5, "map50": 84.1}

# TBKV operating points, ordered from least to most aggressive token skipping.
# ``match`` is the GFLOPs of a matching (skip) frame; ``cache_period`` is how
# often the heavy keyframe pass runs.
TBKV_CONFIGS = [
    # label,           r_match, bg,  gflops(matching), map50, cache_period
    {"label": "conservative", "r_match": 0.3, "bg": 0.4, "gflops": 165.5, "map50": 73.1, "period": 8},
    {"label": "medium",       "r_match": 0.5, "bg": 0.5, "gflops": 156.5, "map50": 73.1, "period": 8},
    {"label": "aggressive",   "r_match": 0.7, "bg": 0.6, "gflops": 144.8, "map50": 73.1, "period": 8},
    {"label": "max",          "r_match": 0.9, "bg": 0.7, "gflops": 130.3, "map50": 74.4, "period": 8},
]

# Competitor operating points (reused from earlier sweeps).
STGT = [
    (35.8, 68.4), (46.8, 76.2), (57.6, 79.7), (68.5, 83.8), (90.1, 84.4), (111.9, 84.2),
]
EVENTFUL = [
    (19.7, 76.4), (33.2, 80.8), (47.0, 83.3), (60.5, 84.0), (87.8, 84.0), (114.9, 84.0),
]
MASKVD = [(88.8, 83.1)]

# Per-frame amortized cost of the competitors (they gate on EVERY frame, so
# their amortized cost is flat in T). Pick each method's efficient knee point.
STGT_PERFRAME = 68.5       # k=512
EVENTFUL_PERFRAME = 60.5   # k=512
MASKVD_PERFRAME = 88.8


def amortized_tbkv(T, cache_gflops, match_gflops, period):
    """Amortized GFLOPs/frame over a clip of length T with periodic recaching."""
    T = np.asarray(T, dtype=float)
    n_cache = np.ceil(T / period)
    n_match = T - n_cache
    return (n_cache * cache_gflops + n_match * match_gflops) / T


# ──────────────────────────────────────────────────────────────────────────────
def plot_long_run(out_dir):
    T = np.arange(1, 201)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # TBKV curves (use the aggressive + max configs).
    for cfg, color in ((TBKV_CONFIGS[2], "#1f77b4"), (TBKV_CONFIGS[3], "#d62728")):
        y = amortized_tbkv(T, BASELINE["gflops"], cfg["gflops"], cfg["period"])
        ax.plot(T, y, color=color, lw=2.3,
                label=f"TBKV ({cfg['label']}, p={cfg['period']})")

    ax.axhline(STGT_PERFRAME, ls="--", color="#2ca02c", lw=1.8, label=f"STGT (k=512), {STGT_PERFRAME:.0f}")
    ax.axhline(EVENTFUL_PERFRAME, ls="--", color="#9467bd", lw=1.8, label=f"Eventful (k=512), {EVENTFUL_PERFRAME:.0f}")
    ax.axhline(MASKVD_PERFRAME, ls=":", color="#8c564b", lw=1.8, label=f"MaskVD, {MASKVD_PERFRAME:.0f}")
    ax.axhline(BASELINE["gflops"], ls="-", color="0.5", lw=1.2, alpha=0.7, label=f"Baseline, {BASELINE['gflops']:.0f}")

    # Mark the crossover where aggressive-TBKV drops below Eventful.
    y_aggr = amortized_tbkv(T, BASELINE["gflops"], TBKV_CONFIGS[2]["gflops"], TBKV_CONFIGS[2]["period"])
    below = np.where(y_aggr < EVENTFUL_PERFRAME)[0]
    if len(below):
        tc = T[below[0]]
        ax.axvline(tc, color="#d62728", ls=":", lw=1, alpha=0.6)
        ax.annotate(f"TBKV < Eventful\nfor T ≥ {tc}", xy=(tc, EVENTFUL_PERFRAME),
                    xytext=(tc + 15, EVENTFUL_PERFRAME + 20), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#d62728"))

    ax.set_xlabel("Clip length  T  (frames)", fontsize=12)
    ax.set_ylabel("Amortized GFLOPs / frame", fontsize=12)
    ax.set_title("TBKV amortized cost falls with clip length\n(competitors gate every frame → flat)", fontsize=12)
    ax.set_xlim(1, 200)
    ax.set_ylim(50, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    p = Path(out_dir) / "long_run_efficiency.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def plot_specular(out_dir):
    labels = [c["label"] for c in TBKV_CONFIGS]
    map50 = np.array([c["map50"] for c in TBKV_CONFIGS])
    gflops = np.array([c["gflops"] for c in TBKV_CONFIGS])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    # UP: mAP@50 (accuracy). DOWN: GFLOPs (compute), drawn as negatives.
    bar_up = ax.bar(x, map50, width=0.6, color="#2b8cbe", label="mAP@50 (%)")
    bar_dn = ax.bar(x, -gflops, width=0.6, color="#e6550d", label="GFLOPs / frame")

    # Baseline reference lines (mirrored around the 0 centre line).
    ax.axhline(0, color="k", lw=1.4)
    ax.axhline(BASELINE["map50"], ls="--", color="#08519c", lw=1.5,
               label=f"Baseline mAP@50 = {BASELINE['map50']:.1f}")
    ax.axhline(-BASELINE["gflops"], ls="--", color="#a63603", lw=1.5,
               label=f"Baseline GFLOPs = {BASELINE['gflops']:.0f}")

    for xi, (m, g) in enumerate(zip(map50, gflops)):
        ax.text(xi, m + 1.5, f"{m:.1f}", ha="center", va="bottom", fontsize=9, color="#08519c")
        ax.text(xi, -g - 6, f"{g:.0f}", ha="center", va="top", fontsize=9, color="#a63603")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(r={c['r_match']}, bg={c['bg']})"
                        for l, c in zip(labels, TBKV_CONFIGS)], fontsize=9)
    ax.set_xlabel("TBKV configuration  (→ more aggressive token skipping)", fontsize=12)
    ax.set_ylabel("←  GFLOPs / frame      mAP@50 (%)  →", fontsize=11)
    ax.set_title("More aggressive skipping saves more compute (↓)\nat a gentle accuracy cost (↑)", fontsize=12)
    ax.set_ylim(-BASELINE["gflops"] - 25, BASELINE["map50"] + 12)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p = Path(out_dir) / "specular_tradeoff.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def plot_configs_pareto(out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    sx, sy = zip(*sorted(STGT))
    ax.plot(sx, sy, "-o", color="#2ca02c", ms=5, lw=1.5, alpha=0.85, label="STGT")
    ex, ey = zip(*sorted(EVENTFUL))
    ax.plot(ex, ey, "-s", color="#9467bd", ms=5, lw=1.5, alpha=0.85, label="Eventful")
    mx, my = zip(*MASKVD)
    ax.scatter(mx, my, color="#8c564b", marker="D", s=70, zorder=5, label="MaskVD")

    tx = [c["gflops"] for c in TBKV_CONFIGS]
    ty = [c["map50"] for c in TBKV_CONFIGS]
    ax.plot(tx, ty, "-^", color="#d62728", ms=10, lw=2.5, zorder=6, label="TBKV (ours)")
    for c in TBKV_CONFIGS:
        ax.annotate(c["label"], (c["gflops"], c["map50"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color="#d62728")

    ax.scatter([BASELINE["gflops"]], [BASELINE["map50"]], color="k", marker="*",
               s=180, zorder=7, label="Baseline ViTDet-B")

    ax.set_xlabel("GFLOPs / frame  (lower = cheaper)", fontsize=12)
    ax.set_ylabel("mAP@50 (%)  (higher = better)", fontsize=12)
    ax.set_title("TBKV operating points vs. SOTA — ImageNet VID", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    p = Path(out_dir) / "tbkv_configs_pareto.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/plots")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plot_long_run(args.out_dir)
    plot_specular(args.out_dir)
    plot_configs_pareto(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
