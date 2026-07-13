#!/usr/bin/env python3
"""Parse the OCTA hyperparameter sweep + compare with stored SOTA baselines.

Reads every results/evaluate/vivit_kinetics400/octa_sweep/<name>/output.txt,
extracts Top-1/Top-5/matching-GFLOPs, reads the config knobs from config.yml,
and writes a Markdown comparison table (SOTA baselines + full OCTA sweep +
best-config recommendation) to results/OCTA_RESULTS.md.

All numbers are n_items=10, frame_split=16, single-CPU inference on the same
ViViT-B / Kinetics-400 weights and eval harness (evaluate_vivit_metrics), so
GFLOPs are directly comparable across methods.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO / "results" / "evaluate" / "vivit_kinetics400" / "octa_sweep"
OUT_MD = REPO / "results" / "OCTA_RESULTS.md"

# Stored SOTA baselines (same harness, n=10, frame_split=16). Source:
# results/TBKV_RESULTS.md section 3 + session baseline-comparison notes.
BASELINES = [
    # name,                       gflops,  top1,  top5
    ("ViViT-B (dense baseline)",  3359.0,  80.0, 100.0),
    ("Eventful (temporal, k=24)",  617.0,  80.0, 100.0),
    ("TBKV",                      1537.0,  70.0, 100.0),
]

_PATS = {
    "top1": re.compile(r"Top-1 Accuracy\s*:\s*([\d.]+)%"),
    "top5": re.compile(r"Top-5 Accuracy\s*:\s*([\d.]+)%"),
    "gflops": re.compile(r"Total GFLOPs \(MATCHING[^:]*:\s*([\d.]+)"),
    "frames": re.compile(r"avg matching frames/video\s*:\s*([\d.]+)"),
}


def _parse_output(path: Path):
    text = path.read_text()
    out = {}
    for key, pat in _PATS.items():
        m = pat.search(text)
        out[key] = float(m.group(1)) if m else None
    return out


def _parse_config(path: Path):
    cfg = yaml.safe_load(path.read_text())
    keys = (
        "cluster_similarity_threshold",
        "merging_iterations",
        "object_match_threshold",
        "offset_window_size",
    )
    return {k: cfg.get(k) for k in keys}


def collect():
    rows = []
    for d in sorted(SWEEP_DIR.iterdir()):
        out_txt = d / "output.txt"
        cfg_yml = d / "config.yml"
        if not (d.is_dir() and out_txt.is_file() and cfg_yml.is_file()):
            continue
        m = _parse_output(out_txt)
        if m["gflops"] is None or m["top1"] is None:
            continue
        c = _parse_config(cfg_yml)
        rows.append({"name": d.name, **m, **c})
    return rows


def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def build_markdown(rows):
    lines = []
    lines.append("# OCTA — Object-Centric Temporal Caching: Results & Hyperparameter Sweep\n")
    lines.append(
        "ViViT-B on Kinetics-400 (val). All rows use the **same** eval harness "
        "(`evaluate_vivit_metrics`), `n_items=10`, `frame_split=16`, single-CPU "
        "inference and identical weights, so GFLOPs are directly comparable.\n"
    )
    lines.append("> GFLOPs = total cost of the **matching** pass per video "
                 "(sum of counted linear+matmul+add+bias FLOPs). Accuracy is noisy "
                 "at n=10 (one clip = 10 %); relative accuracy/FLOP trends are the point.\n")

    # ---- Section 1: OCTA vs SOTA ------------------------------------------
    lines.append("\n## 1. OCTA vs. SOTA\n")
    lines.append("| Method | GFLOPs / video | Top-1 (%) | Top-5 (%) | FLOP reduction |")
    lines.append("|---|---:|---:|---:|---:|")
    base = 3359.0
    for name, gf, t1, t5 in BASELINES:
        red = "—" if abs(gf - base) < 1e-6 else f"−{(1 - gf / base) * 100:.0f}%"
        lines.append(f"| {name} | {gf:.0f} | {t1:.1f} | {t5:.1f} | {red} |")
    best = pick_best(rows)
    if best is not None:
        red = f"−{(1 - best['gflops'] / base) * 100:.0f}%"
        lines.append(
            f"| **OCTA (best: {best['name']})** | {best['gflops']:.0f} | "
            f"{best['top1']:.1f} | {best['top5']:.1f} | **{red}** |"
        )

    # ---- Section 2: full sweep -------------------------------------------
    lines.append("\n## 2. OCTA hyperparameter sweep\n")
    lines.append("Knobs: `thr` = cluster_similarity_threshold, `iters` = merging_iterations, "
                 "`match` = object_match_threshold, `win` = offset_window_size.\n")
    lines.append("| Config | thr | iters | match | win | GFLOPs | Top-1 (%) | Top-5 (%) | match frames |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: r["gflops"]):
        star = " ⭐" if best is not None and r["name"] == best["name"] else ""
        lines.append(
            f"| {r['name']}{star} | {_fmt(r['cluster_similarity_threshold'])} | "
            f"{_fmt(r['merging_iterations'], 0)} | {_fmt(r['object_match_threshold'])} | "
            f"{_fmt(r['offset_window_size'], 1)} | {r['gflops']:.1f} | "
            f"{r['top1']:.1f} | {r['top5']:.1f} | {_fmt(r['frames'], 1)} |"
        )

    # ---- Section 3: recommendation ---------------------------------------
    lines.append("\n## 3. Best configuration\n")
    if best is not None:
        lines.append(
            f"**{best['name']}** — Top-1 {best['top1']:.1f}% / Top-5 {best['top5']:.1f}% "
            f"at {best['gflops']:.1f} GFLOPs/video "
            f"(−{(1 - best['gflops'] / base) * 100:.0f}% vs dense).\n"
        )
        lines.append(
            f"- cluster_similarity_threshold: {best['cluster_similarity_threshold']}\n"
            f"- merging_iterations: {best['merging_iterations']}\n"
            f"- object_match_threshold: {best['object_match_threshold']}\n"
            f"- offset_window_size: {best['offset_window_size']}\n"
        )
    lines.append("\n_Generated by `scripts/misc/parse_octa_sweep.py`._\n")
    return "\n".join(lines)


def pick_best(rows):
    """Best = highest Top-1, then highest Top-5, then lowest GFLOPs (Pareto pick)."""
    if not rows:
        return None
    return sorted(rows, key=lambda r: (-r["top1"], -r["top5"], r["gflops"]))[0]


def main():
    rows = collect()
    if not rows:
        print("No completed sweep runs found yet in", SWEEP_DIR)
        return
    md = build_markdown(rows)
    OUT_MD.write_text(md)
    print(f"Parsed {len(rows)} configs -> {OUT_MD}")
    print(md)


if __name__ == "__main__":
    main()
