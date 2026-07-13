#!/usr/bin/env python3
"""
Plot accuracy vs. GFLOPs tradeoff for all methods on ImageNet VID.

Usage:
    cd /home/cc/TBKV
    python scripts/plot_vid_comparison.py [--results_dir /dev/shm/compare/n10]

The script reads output.txt files from each method's result directory and
extracts mAP@50 and GFLOPs/frame to produce a Pareto-frontier plot.

Output: scripts/plots/accuracy_vs_gflops.pdf
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colour / marker palette ──────────────────────────────────────────────
STYLES = {
    "Baseline":    {"color": "black",  "marker": "D", "ls": "--", "label": "ViTDet-B (baseline)"},
    "STGT":        {"color": "blue",   "marker": "s", "ls": "-",  "label": "STGT"},
    "Eventful":    {"color": "green",  "marker": "^", "ls": "-",  "label": "Eventful-transformer"},
    "Spatiotempl": {"color": "purple", "marker": "v", "ls": "-",  "label": "Spatiotemporal Eventful"},
    "MaskVD":      {"color": "orange", "marker": "o", "ls": "-",  "label": "MaskVD"},
    "TBKV":        {"color": "red",    "marker": "*", "ls": "-",  "label": "TBKV (ours)", "zorder": 10, "s": 200},
}

# ── Parsing helpers ───────────────────────────────────────────────────────

def _parse_float(text, key):
    """Extract a float after 'key:' in text."""
    m = re.search(rf"{re.escape(key)}\s*[:=]\s*([\d.e+\-]+)", text)
    return float(m.group(1)) if m else None


def _parse_counts_section(text):
    """Return a dict of {key: value} from a 'Counts' section in output.txt."""
    counts = {}
    in_counts = False
    for line in text.splitlines():
        if line.strip() == "Counts":
            in_counts = True
            continue
        if in_counts:
            m = re.match(r"\s+([\w_]+):\s+([\d.e+\-]+)", line)
            if m:
                counts[m.group(1)] = float(m.group(2))
            elif line.strip() == "" or re.match(r"\w", line):
                in_counts = False
    return counts


def _total_gflops(counts):
    """Sum all FLOPs categories and convert to GFLOPs."""
    return sum(counts.values()) / 1e9


def parse_eventful_output(output_txt):
    """Parse one or more 'Token top k=…' sections from output.txt.

    Returns list of (gflops, map50) tuples.
    """
    points = []
    sections = re.split(r"Token top k=\d+", output_txt)
    headers = re.findall(r"Token top k=(\d+)", output_txt)

    for k_str, section in zip(headers, sections[1:]):
        map50 = _parse_float(section, "map_50")
        counts = _parse_counts_section(section)
        if map50 is not None and counts:
            gf = _total_gflops(counts)
            points.append((gf, float(map50)))

    return points


def parse_vanilla_output(output_txt):
    """Parse a 'Vanilla' evaluation section."""
    m_map = re.search(r"map_50:\s*([\d.e+\-]+)", output_txt)
    counts = _parse_counts_section(output_txt)
    if m_map and counts:
        return [(_total_gflops(counts), float(m_map.group(1)))]
    return []


def parse_maskvd_txt(results_txt):
    """Parse MaskVD results.txt (custom format)."""
    map50 = _parse_float(results_txt, "mAP@50")
    gf = _parse_float(results_txt, "GFLOPs/frame")
    if map50 is not None and gf is not None:
        return [(gf, map50)]
    return []


# ── Load all results ─────────────────────────────────────────────────────

def load_results(results_dir):
    """Discover and load all result files, return dict[method_key] = [(gf, map50)]."""
    d = Path(results_dir)
    data = {}

    method_map = {
        "baseline": ("Baseline", parse_vanilla_output),
        "stgt": ("STGT", parse_eventful_output),
        "temporal": ("Eventful", parse_eventful_output),
        "spatiotemporal": ("Spatiotempl", parse_eventful_output),
        "maskvd": ("MaskVD", None),   # uses results.txt
        "tbkv": ("TBKV", parse_vanilla_output),   # single "Evaluation" section
    }

    for folder in sorted(d.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        for key, (method_name, parser) in method_map.items():
            if key in name:
                output_file = folder / "output.txt"
                results_file = folder / "results.txt"
                if results_file.exists() and method_name == "MaskVD":
                    pts = parse_maskvd_txt(results_file.read_text())
                elif output_file.exists() and parser is not None:
                    pts = parser(output_file.read_text())
                else:
                    pts = []
                if pts:
                    if method_name not in data:
                        data[method_name] = []
                    data[method_name].extend(pts)
                break

    return data


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_accuracy_vs_gflops(data, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    legend_handles = []

    for method_key, style in STYLES.items():
        pts = data.get(method_key, [])
        if not pts:
            continue

        # Sort by GFLOPs for curve methods
        pts = sorted(pts, key=lambda p: p[0])
        gf = [p[0] for p in pts]
        mp = [p[1] * 100 for p in pts]   # convert to percent

        color = style["color"]
        marker = style["marker"]
        label = style["label"]
        ls = style.get("ls", "-")
        zorder = style.get("zorder", 5)
        sz = style.get("s", 80)

        if len(pts) == 1:
            ax.scatter(gf, mp, color=color, marker=marker, s=sz,
                       zorder=zorder, edgecolors="white", linewidths=0.8)
        else:
            ax.plot(gf, mp, color=color, marker=marker, ls=ls,
                    linewidth=1.5, markersize=7, zorder=zorder)

        legend_handles.append(
            mpatches.Patch(color=color, label=label)
        )

    ax.set_xlabel("GFLOPs / frame", fontsize=12)
    ax.set_ylabel("mAP@50 (%)", fontsize=12)
    ax.set_title("Accuracy vs. Efficiency on ImageNet VID (672px)", fontsize=13)
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_efficiency_curve(data, out_path):
    """Bar-chart comparing best mAP@50 per method at similar GFLOPs budgets."""
    target_gflops = [20, 40, 60, 80, 100]   # GFLOPs budgets

    methods = list(STYLES.keys())
    x = np.arange(len(target_gflops))
    width = 0.15
    colors = [STYLES[m]["color"] for m in methods if m in data]

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_i = 0
    for m_idx, method_key in enumerate(methods):
        pts = data.get(method_key, [])
        if not pts:
            continue

        pts = sorted(pts, key=lambda p: p[0])
        gf_arr = np.array([p[0] for p in pts])
        mp_arr = np.array([p[1] * 100 for p in pts])

        # Interpolate at target GFLOPs
        bar_vals = []
        for tg in target_gflops:
            # Find the closest result at or below target GFLOPs
            mask = gf_arr <= tg
            if mask.sum() > 0:
                best_idx = np.argmax(mp_arr[mask])
                bar_vals.append(mp_arr[mask][best_idx])
            else:
                bar_vals.append(0.0)

        label = STYLES[method_key]["label"]
        color = STYLES[method_key]["color"]
        ax.bar(x + bar_i * width, bar_vals, width, label=label, color=color, alpha=0.85)
        bar_i += 1

    ax.set_xlabel("GFLOPs Budget / frame", fontsize=12)
    ax.set_ylabel("mAP@50 (%)", fontsize=12)
    ax.set_title("Best mAP@50 at Different Efficiency Budgets", fontsize=13)
    ax.set_xticks(x + width * bar_i / 2)
    ax.set_xticklabels([f"≤{tg}" for tg in target_gflops])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def print_comparison_table(data):
    """Print a LaTeX-friendly comparison table."""
    print("\n" + "="*70)
    print("COMPARISON TABLE  (n_items=10 on ImageNet VID 672px)")
    print("="*70)
    print(f"{'Method':<30} {'GFLOPs/frame':>14} {'mAP@50 (%)':>12}")
    print("-"*70)

    for method_key, style in STYLES.items():
        pts = data.get(method_key, [])
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: p[0])
        label = style["label"]

        if len(pts) == 1:
            gf, mp = pts[0]
            print(f"{label:<30} {gf:>14.1f} {mp*100:>12.1f}")
        else:
            # Print best accuracy point and most efficient point
            best_acc = max(pts, key=lambda p: p[1])
            most_eff = min(pts, key=lambda p: p[0])
            print(f"{label:<30} (sweep over {len(pts)} k values)")
            print(f"  {'best accuracy':<28} {best_acc[0]:>14.1f} {best_acc[1]*100:>12.1f}")
            print(f"  {'most efficient':<28} {most_eff[0]:>14.1f} {most_eff[1]*100:>12.1f}")

    print("="*70)
    print("\nLaTeX table row format:")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Method & GFLOPs/frame & mAP@50 (\%) \\")
    print(r"\midrule")
    for method_key, style in STYLES.items():
        pts = data.get(method_key, [])
        if not pts:
            continue
        label = style["label"]
        if len(pts) == 1:
            gf, mp = pts[0]
            print(rf"{label} & {gf:.1f} & {mp*100:.1f} \\")
        else:
            best_acc = max(pts, key=lambda p: p[1])
            print(rf"{label} & {best_acc[0]:.1f} & {best_acc[1]*100:.1f} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Accuracy-efficiency tradeoff on ImageNet VID.}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="/dev/shm/compare/n10",
                        help="Directory containing per-method result subdirectories")
    parser.add_argument("--out_dir", default="scripts/plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run scripts/compare_all_vid.py first.")
        return

    data = load_results(results_dir)

    if not data:
        print(f"No results found in {results_dir}")
        print("Expected subdirectories: baseline_672/, stgt_672/, temporal_672/, maskvd_672/, tbkv_672/")
        return

    print(f"Loaded results for: {list(data.keys())}")
    print_comparison_table(data)

    out_dir = args.out_dir
    plot_accuracy_vs_gflops(data, f"{out_dir}/accuracy_vs_gflops_vid.pdf")
    plot_efficiency_curve(data, f"{out_dir}/efficiency_budget_vid.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
