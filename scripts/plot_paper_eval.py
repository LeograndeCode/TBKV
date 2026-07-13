#!/usr/bin/env python3
"""
Paper-ready evaluation figures for the TBKV study.

All numbers are validated n_items=10 runs on a single Quadro RTX 6000, using the
identical ViTDet-B / ViViT-B weights across every method. FLOPs use the in-repo
CountedLinear / matmul counters; latency and peak memory use CUDA events with an
identical warm-up + reset_peak_memory harness across all methods.

Figures produced (clean, large fonts, minimal text):
  1. fig_pareto_vid.pdf/png     accuracy vs. compute frontier (ImageNet VID)
  2. fig_latency_memory_vid.*   wall-clock latency and peak memory (ImageNet VID)
  3. fig_pareto_vivit.*         accuracy vs. compute frontier (Kinetics-400)

Edit the dictionaries below and re-run to regenerate after new evaluations.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Global paper style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

C = {
    "baseline": "#000000",
    "eventful": "#7A5FB0",
    "stgt":     "#2E8B57",
    "maskvd":   "#8C564B",
    "tbkv":     "#D62728",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA  (ImageNet VID, ViTDet-B, 672 px, n=10)
# ══════════════════════════════════════════════════════════════════════════════
BASELINE_VID = {"gflops": 174.5, "map50": 84.1, "latency": 69.6, "memory": 760}

# Full efficiency sweeps of the competitors (GFLOPs/frame, mAP@50 %).
STGT_SWEEP = [(35.8, 68.4), (46.8, 76.2), (57.6, 79.7),
              (68.5, 83.8), (90.1, 84.4), (111.9, 84.2)]
EVENTFUL_SWEEP = [(19.7, 76.4), (33.2, 80.8), (47.0, 83.3),
                  (60.5, 84.0), (87.8, 84.0), (114.9, 84.0)]
MASKVD_POINT = (88.8, 83.1)

# TBKV operating points (GFLOPs/frame, mAP@50 %).
TBKV_POINTS = [
    {"label": "TBKV",             "gflops": 156.5, "map50": 72.0},
    {"label": "TBKV (aggr.)",     "gflops": 130.3, "map50": 74.4},
    {"label": "TBKV + 2nd",       "gflops": 132.0, "map50": 70.5},
    {"label": "TBKV all-blocks",  "gflops": 125.7, "map50": 73.5},
]

# TBKV token-skip sweep, ordered by increasing skipping aggressiveness
# (label, r_match, GFLOPs/frame, mAP@50 %).
TBKV_SWEEP = [
    ("conservative", 0.3, 165.5, 73.1),
    ("medium",       0.5, 156.5, 73.1),
    ("aggressive",   0.7, 144.8, 73.1),
    ("max",          0.9, 130.3, 74.4),
]

# Amortization: keyframe (caching) pass runs every CACHE_PERIOD frames.
CACHE_PERIOD = 8
KEYFRAME_GFLOPS = 174.5           # heavy keyframe pass cost
FLAT_COMPETITORS = [              # steady per-frame cost (gate every frame)
    ("Baseline", 174.5, C["baseline"], "-"),
    ("MaskVD",   88.8,  C["maskvd"],   ":"),
    ("STGT",     68.5,  C["stgt"],     "--"),
    ("Eventful", 60.5,  C["eventful"], "--"),
]

# Latency + peak-memory triage (single method on GPU, identical harness).
#   name, GFLOPs, latency ms, memory MB, mAP@50, color
TRIAGE = [
    ("Baseline", 174.5, 69.6,  760, 84.1, C["baseline"]),
    ("Eventful", 60.5,  58.3, 2118, 84.0, C["eventful"]),
    ("STGT",     68.5,  56.2, 1242, 83.8, C["stgt"]),
    ("MaskVD",   88.8,  56.0,  927, 83.1, C["maskvd"]),
    ("TBKV",     156.5, 101.8, 929, 72.0, C["tbkv"]),
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA  (Kinetics-400, ViViT-B, per-frame)
# ══════════════════════════════════════════════════════════════════════════════
VIVIT = [
    # name, GFLOPs/frame, top-1 %, color, marker
    ("Baseline", 14.74, 80.0, C["baseline"], "*"),
    ("Eventful", 2.70,  80.0, C["eventful"], "s"),
    ("TBKV",     7.25,  60.0, C["tbkv"],     "^"),
]


# ══════════════════════════════════════════════════════════════════════════════
def plot_pareto_vid(out_dir):
    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    sx, sy = zip(*sorted(STGT_SWEEP))
    ax.plot(sx, sy, "-o", color=C["stgt"], ms=6, lw=2, label="STGT")
    ex, ey = zip(*sorted(EVENTFUL_SWEEP))
    ax.plot(ex, ey, "-s", color=C["eventful"], ms=6, lw=2, label="Eventful")
    ax.scatter(*MASKVD_POINT, color=C["maskvd"], marker="D", s=90,
               zorder=5, label="MaskVD")

    tx = [p["gflops"] for p in TBKV_POINTS]
    ty = [p["map50"] for p in TBKV_POINTS]
    ax.scatter(tx, ty, color=C["tbkv"], marker="^", s=130, zorder=6,
               edgecolor="white", linewidth=0.8, label="TBKV (ours)")

    ax.scatter(BASELINE_VID["gflops"], BASELINE_VID["map50"], color=C["baseline"],
               marker="*", s=320, zorder=7, label="Baseline")

    # Shade the efficient region the competitors occupy.
    ax.axvspan(0, 95, color="#2E8B57", alpha=0.05)
    ax.text(48, 69.5, "efficient frontier", color=C["stgt"], fontsize=11,
            style="italic", ha="center")

    ax.set_xlabel("GFLOPs / frame  (lower is better)")
    ax.set_ylabel("mAP@50 (%)  (higher is better)")
    ax.set_title("Accuracy vs. compute — ImageNet VID")
    ax.set_xlim(0, 185)
    ax.set_ylim(67, 85.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.72), frameon=True)
    _save(fig, out_dir, "fig_pareto_vid")


def plot_latency_memory_vid(out_dir):
    names = [t[0] for t in TRIAGE]
    lat = [t[2] for t in TRIAGE]
    mem = [t[3] for t in TRIAGE]
    cols = [t[5] for t in TRIAGE]
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.bar(x, lat, color=cols, width=0.62)
    ax1.set_ylabel("Latency / frame (ms)")
    ax1.set_title("Wall-clock latency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylim(0, max(lat) * 1.18)
    for xi, v in zip(x, lat):
        ax1.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=11)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.bar(x, mem, color=cols, width=0.62)
    ax2.set_ylabel("Peak memory (MB)")
    ax2.set_title("Peak GPU memory")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylim(0, max(mem) * 1.18)
    for xi, v in zip(x, mem):
        ax2.text(xi, v + 35, f"{v:.0f}", ha="center", va="bottom", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Deployment cost — ImageNet VID (TBKV is slowest; no memory win)",
                 fontsize=14, y=1.02)
    _save(fig, out_dir, "fig_latency_memory_vid")


def plot_pareto_vivit(out_dir):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for name, g, acc, col, mk in VIVIT:
        ax.scatter(g, acc, color=col, marker=mk,
                   s=260 if mk == "*" else 150, zorder=5,
                   edgecolor="white", linewidth=0.8, label=name)
        ax.annotate(f"{name}: {g:.1f} GF, {acc:.0f}%", (g, acc),
                    textcoords="offset points", xytext=(8, 8), fontsize=10)

    ax.set_xlabel("GFLOPs / frame  (lower is better)")
    ax.set_ylabel("Top-1 accuracy (%)  (higher is better)")
    ax.set_title("Accuracy vs. compute — Kinetics-400")
    ax.set_xlim(0, 17)
    ax.set_ylim(55, 85)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, out_dir, "fig_pareto_vivit")


def _save(fig, out_dir, stem):
    fig.tight_layout()
    p = Path(out_dir) / f"{stem}.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved: {p}")


def plot_specular(out_dir):
    labels = [s[0] for s in TBKV_SWEEP]
    rmatch = [s[1] for s in TBKV_SWEEP]
    gflops = np.array([s[2] for s in TBKV_SWEEP])
    map50 = np.array([s[3] for s in TBKV_SWEEP])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # Bars up = accuracy retained; bars down = compute spent.
    ax.bar(x, map50, width=0.6, color="#2B8CBE", label="mAP@50 (%)")
    ax.bar(x, -gflops, width=0.6, color="#E6550D", label="GFLOPs / frame")

    ax.axhline(0, color="k", lw=1.3)
    ax.axhline(BASELINE_VID["map50"], ls="--", color="#08519C", lw=1.4,
               label=f"Baseline mAP@50 = {BASELINE_VID['map50']:.1f}")
    ax.axhline(-BASELINE_VID["gflops"], ls="--", color="#A63603", lw=1.4,
               label=f"Baseline GFLOPs = {BASELINE_VID['gflops']:.0f}")

    for xi, (m, g) in enumerate(zip(map50, gflops)):
        ax.text(xi, m + 1.5, f"{m:.1f}", ha="center", va="bottom",
                fontsize=11, color="#08519C")
        ax.text(xi, -g - 6, f"{g:.0f}", ha="center", va="top",
                fontsize=11, color="#A63603")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(r={r})" for l, r in zip(labels, rmatch)])
    ax.set_xlabel("TBKV token-skip setting  (more aggressive \u2192)")
    ax.set_ylabel("\u2190  GFLOPs / frame        mAP@50 (%)  \u2192")
    ax.set_title("Compute (\u2193) and accuracy (\u2191) across TBKV settings", pad=28)
    ax.set_ylim(-BASELINE_VID["gflops"] - 28, BASELINE_VID["map50"] + 30)
    ax.legend(loc="upper center", ncol=2, frameon=True,
              bbox_to_anchor=(0.5, 1.10))
    ax.grid(True, axis="y", alpha=0.2)
    _save(fig, out_dir, "fig_specular_tradeoff")


def _amortized(T, keyframe, match, period):
    T = np.asarray(T, dtype=float)
    n_cache = np.ceil(T / period)
    n_match = T - n_cache
    return (n_cache * keyframe + n_match * match) / T


def plot_long_run(out_dir):
    T = np.arange(1, 201)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    for (label, r, g, m), color in zip(
            (TBKV_SWEEP[2], TBKV_SWEEP[3]), ("#1F77B4", C["tbkv"])):
        y = _amortized(T, KEYFRAME_GFLOPS, g, CACHE_PERIOD)
        ax.plot(T, y, color=color, lw=2.4, label=f"TBKV ({label})")

    for name, val, color, ls in FLAT_COMPETITORS:
        ax.axhline(val, ls=ls, color=color, lw=1.7,
                   label=f"{name} ({val:.0f})")

    ax.set_xlabel("Clip length  T  (frames)")
    ax.set_ylabel("Amortized GFLOPs / frame")
    ax.set_title("Amortized compute vs. clip length")
    ax.set_xlim(1, 200)
    ax.set_ylim(50, 180)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2, frameon=True)
    _save(fig, out_dir, "fig_long_run_efficiency")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/plots")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plot_pareto_vid(args.out_dir)
    plot_latency_memory_vid(args.out_dir)
    plot_pareto_vivit(args.out_dir)
    plot_specular(args.out_dir)
    plot_long_run(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
