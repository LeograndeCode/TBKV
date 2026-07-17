#!/usr/bin/env python3
"""
Gains histogram for the TempoMem paper: Eventful, TempoMem with raw cache
(standalone, unmerged prototype cache) and Eventful+TBKV (= TempoMem) against
the dense ViViT-B / Kinetics-400 baseline.

Two panels (never a dual axis): steady-state compute per matching frame, and
Top-1 accuracy. Accounting: matching-frame GFLOPs divided by the number of
matching frames — the honest steady-state cost once warm-up has amortized.

Numbers are read from the result directories listed in RUNS below; each entry
records its provenance (FULL = full validation split, PRELIM = subset run).
Re-run this script after the full-scale queue lands to refresh the figure.

Output: paper/Figures/fig_gains_hist.{png,pdf} + results/comparison copy.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
VIVIT_RESULTS = REPO / "results" / "evaluate" / "vivit_kinetics400"

# Categorical palette (dataviz reference instance, light mode, fixed slot
# order 1..3); the dense baseline is a neutral gray, not a series hue.
GRAY = "#6b6a66"
BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
INK = "#0b0b0b"
INK_2 = "#52514e"


def parse_eventful_style(output_txt):
    """eventful_tbkv_vivit output: Top-1 % + 'caching A + matching B' GF/clip.
    16-frame clips, 4 caching frames -> 12 matching frames."""
    text = Path(output_txt).read_text()
    top1 = float(re.findall(r"Top-1 Accuracy\s*:\s*([\d.]+)%", text)[-1])
    match_gf = float(
        re.findall(r"caching\s+[\d.]+\s*\+\s*matching\s+([\d.]+)", text)[-1]
    )
    return top1, match_gf / 12.0


def parse_standalone_style(output_txt):
    """tbkv_vivit standalone output: matching GF/video + matching frames/video."""
    text = Path(output_txt).read_text()
    top1 = float(re.findall(r"Top-1 Accuracy\s*:\s*([\d.]+)%", text)[-1])
    match_gf = float(
        re.findall(r"Total GFLOPs \(MATCHING = reported metric\)\s*:\s*([\d.]+)",
                   text)[-1]
    )
    frames = float(
        re.findall(r"avg matching frames/video\s*:\s*([\d.]+)", text)[-1]
    )
    return top1, match_gf / frames


def parse_dense_style(output_txt, frames_per_clip=16):
    """vivit_kinetics400 base output (run_evaluations): top_1 fraction +
    Counts flops keys per clip; every frame costs the same."""
    text = Path(output_txt).read_text()
    top1 = float(re.findall(r"top_1:\s*([\d.]+)", text)[-1]) * 100.0
    flops = sum(
        float(v) for v in re.findall(r"_flops:\s*([\d.e+]+)", text)
    )
    return top1, flops / 1e9 / frames_per_clip


def first_existing(candidates):
    for path, tag in candidates:
        if Path(path).exists():
            return Path(path), tag
    return None, None


# (label, color, parser, [(output.txt candidates, provenance tag)])
RUNS = [
    ("Base\n(dense)", GRAY, parse_dense_style, [
        (VIVIT_RESULTS / "base" / "output.txt", "FULL"),
        (VIVIT_RESULTS / "base-n_items=100" / "output.txt", "PRELIM 100"),
    ]),
    ("Eventful", BLUE, parse_eventful_style, [
        (VIVIT_RESULTS / "eventful_tbkv_24" / "output.txt", "FULL"),
    ]),
    ("TempoMem\n(raw cache)", GREEN, parse_standalone_style, [
        (VIVIT_RESULTS / "tbkv_raw-weights=weights/vivit_b_kinetics400.pth"
         / "output.txt", "FULL"),
        (VIVIT_RESULTS
         / "tbkv_raw-n_items=100-weights=weights/vivit_b_kinetics400.pth"
         / "output.txt", "PRELIM 100"),
    ]),
    ("TempoMem", MAGENTA, parse_eventful_style, [
        (VIVIT_RESULTS / "eventful_tbkv" / "output.txt", "FULL"),
    ]),
]


def main():
    labels, colors, top1s, gflops, tags = [], [], [], [], []
    for label, color, parser, candidates in RUNS:
        path, tag = first_existing(candidates)
        if path is None:
            print(f"WARNING: no results yet for {label!r}; skipping")
            continue
        top1, gf = parser(path)
        labels.append(label)
        colors.append(color)
        top1s.append(top1)
        gflops.append(gf)
        tags.append(tag)
        print(f"{label.replace(chr(10), ' '):<24} {tag:<10} "
              f"top1={top1:5.1f}%  {gf:7.1f} GF/frame   ({path})")

    if len(labels) < 2:
        sys.exit("Not enough runs found to plot.")

    base_gf = gflops[0]
    base_top1 = top1s[0]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), dpi=150,
        gridspec_kw={"wspace": 0.32},
    )
    x = range(len(labels))

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8, color=INK)

    # ── Panel A: steady-state compute ────────────────────────────────────
    bars = ax1.bar(x, gflops, width=0.55, color=colors, zorder=3)
    ax1.set_ylabel("GFLOPs / frame (steady state)", fontsize=8, color=INK)
    for i, (bar, gf) in enumerate(zip(bars, gflops)):
        ax1.annotate(f"{gf:.1f}", (bar.get_x() + bar.get_width() / 2, gf),
                     ha="center", va="bottom", fontsize=8, color=INK,
                     xytext=(0, 2), textcoords="offset points")
        if i > 0:
            ax1.annotate(f"×{base_gf / gf:.1f} less",
                         (bar.get_x() + bar.get_width() / 2, gf),
                         ha="center", va="bottom", fontsize=7.5,
                         color=INK_2, fontweight="bold" if i == len(bars) - 1
                         else "normal",
                         xytext=(0, 13), textcoords="offset points")
    ax1.set_title("Compute per matching frame", fontsize=9, color=INK)

    # ── Panel B: accuracy ────────────────────────────────────────────────
    bars = ax2.bar(x, top1s, width=0.55, color=colors, zorder=3)
    ax2.set_ylabel("Top-1 (%)", fontsize=8, color=INK)
    ax2.set_ylim(0, 100)
    for i, (bar, t1) in enumerate(zip(bars, top1s)):
        text = f"{t1:.1f}" if i == 0 else f"{t1:.1f}\n({t1 - base_top1:+.1f})"
        ax2.annotate(text, (bar.get_x() + bar.get_width() / 2, t1),
                     ha="center", va="bottom", fontsize=7.5, color=INK,
                     xytext=(0, 2), textcoords="offset points")
    ax2.set_title("Top-1 accuracy", fontsize=9, color=INK)

    prelim = [f"{l.replace(chr(10), ' ')}: {t}" for l, t in zip(labels, tags)
              if "PRELIM" in t]
    note = ("ViViT-B / Kinetics-400, matching-frame accounting."
            + ("  Preliminary: " + "; ".join(prelim) if prelim else ""))
    fig.suptitle("Efficiency gains over the dense baseline", fontsize=10,
                 color=INK, y=1.04)
    fig.text(0.01, -0.06, note, fontsize=6.5, color=INK_2)

    out_paper = REPO / "paper" / "Figures" / "fig_gains_hist"
    out_cmp = REPO / "results" / "comparison" / "fig_gains_hist"
    for stem in (out_paper, out_cmp):
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{out_paper}.pdf", bbox_inches="tight")
    print(f"Saved {out_paper}.png/.pdf and {out_cmp}.png")


if __name__ == "__main__":
    main()
