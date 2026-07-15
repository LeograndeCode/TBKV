#!/usr/bin/env python3
"""Hyperparameter sweep for TBKV on ViViT / Kinetics-400.

Each configuration is a separate subprocess invocation of the normal
evaluation entrypoint, so what the sweep measures is exactly what a manual
run would produce. Results are parsed from stdout and written to a CSV that
is appended incrementally -- a crashed or killed sweep keeps everything it
already measured.

Usage:
    python scripts/sweep_tbkv.py --stage 1
    python scripts/sweep_tbkv.py --stage 2 --r-match 0.8 --local-merge-ratio 0.1
    python scripts/sweep_tbkv.py --grid r_match=0.5,0.8 bg_ratio=0.4,0.5
"""

import argparse
import csv
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "scripts" / "evaluate" / "tbkv_vivit_kinetics400.py"

# The caching window is fixed at 4 frames; every remaining frame is a matching
# frame. This is the regime the method is meant to operate in, so it is held
# constant across the sweep rather than being tuned.
FRAME_SPLIT = 4

TOP1_RE = re.compile(r"Top-1 Accuracy\s*:\s*([\d.]+)%")
TOP5_RE = re.compile(r"Top-5 Accuracy\s*:\s*([\d.]+)%")
GFLOPS_RE = re.compile(r"Total GFLOPs \(MATCHING[^)]*\)\s*:\s*([\d.]+)")
# Pair each block header with its token count, so spatial and temporal blocks
# can be told apart (they are printed in one flat list).
TOKENS_RE = re.compile(
    r"\[(\w+)_model\.backbone\.blocks\.(\d+)\][^\[]*?"
    r"avg final tokens \(fg\+bg\+cached\)\s*:\s*([\d.]+)",
    re.S,
)


def run_config(params, n_items, timeout=3600):
    """Run one evaluation; return parsed metrics (or None on failure)."""
    overrides = [f"{k}={v}" for k, v in params.items()]
    cmd = [
        sys.executable, str(EVAL), "tbkv",
        f"n_items={n_items}", f"frame_split={FRAME_SPLIT}", *overrides,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s", flush=True)
        return None

    out = proc.stdout + proc.stderr
    top1, top5 = TOP1_RE.search(out), TOP5_RE.search(out)
    if proc.returncode != 0 or not top1:
        tail = "\n".join(out.strip().splitlines()[-6:])
        print(f"    FAILED (rc={proc.returncode})\n{tail}", flush=True)
        return None

    gflops = GFLOPS_RE.search(out)
    # Spatial-block token counts in block order. The last one is what actually
    # reaches the classifier, and is the number to watch for token collapse.
    spatial_tokens = [
        float(n) for stack, _, n in TOKENS_RE.findall(out)
        if stack == "spatial" and float(n) > 0
    ]

    return {
        "top_1": float(top1.group(1)),
        "top_5": float(top5.group(1)) if top5 else float("nan"),
        "gflops": float(gflops.group(1)) if gflops else float("nan"),
        "tokens_first": spatial_tokens[0] if spatial_tokens else 0.0,
        "tokens_last": spatial_tokens[-1] if spatial_tokens else 0.0,
        "seconds": round(time.time() - t0, 1),
    }


def sweep(grid, n_items, out_csv):
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    print(f"{len(combos)} configurations x {n_items} videos -> {out_csv}\n", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = keys + ["top_1", "top_5", "gflops", "tokens_first", "tokens_last", "seconds"]
    new_file = not out_csv.exists()
    results = []

    with out_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()

        for i, params in enumerate(combos, 1):
            desc = "  ".join(f"{k}={v}" for k, v in params.items())
            print(f"[{i:>2}/{len(combos)}] {desc}", flush=True)
            res = run_config(params, n_items)
            if res is None:
                continue
            row = {**params, **res}
            writer.writerow(row)
            fh.flush()
            results.append(row)
            print(
                f"    top1={res['top_1']:.1f}%  top5={res['top_5']:.1f}%  "
                f"GFLOPs={res['gflops']:.0f}  tokens {res['tokens_first']:.0f}->"
                f"{res['tokens_last']:.0f}  ({res['seconds']:.0f}s)",
                flush=True,
            )

    return results


def report(results):
    if not results:
        print("\nNo successful runs.")
        return
    print("\n" + "=" * 78)
    print("RANKED BY TOP-1")
    print("=" * 78)
    for r in sorted(results, key=lambda r: -r["top_1"]):
        cfg = "  ".join(
            f"{k}={v}" for k, v in r.items()
            if k not in {"top_1", "top_5", "gflops", "tokens_first", "tokens_last", "seconds"}
        )
        print(f"  top1={r['top_1']:5.1f}%  top5={r['top_5']:5.1f}%  "
              f"GFLOPs={r['gflops']:7.0f}   {cfg}")

    # A config is on the Pareto front if nothing else is both more accurate and
    # cheaper. These are the only configurations worth reporting in a paper.
    print("\nPARETO FRONT (accuracy vs GFLOPs)")
    front = [
        r for r in results
        if not any(o["top_1"] > r["top_1"] and o["gflops"] < r["gflops"] for o in results)
    ]
    for r in sorted(front, key=lambda r: r["gflops"]):
        cfg = "  ".join(
            f"{k}={v}" for k, v in r.items()
            if k not in {"top_1", "top_5", "gflops", "tokens_first", "tokens_last", "seconds"}
        )
        print(f"  GFLOPs={r['gflops']:7.0f}  top1={r['top_1']:5.1f}%   {cfg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "sweeps" / "tbkv_sweep.csv")
    ap.add_argument("--r-match", type=float, help="stage 2: fixed r_match from stage 1")
    ap.add_argument("--local-merge-ratio", type=float, help="stage 2: fixed ratio from stage 1")
    ap.add_argument("--grid", nargs="*", help="custom grid, e.g. r_match=0.5,0.8")
    args = ap.parse_args()

    if args.grid:
        grid = {}
        for item in args.grid:
            k, v = item.split("=", 1)
            grid[k] = [float(x) if "." in x else int(x) for x in v.split(",")]
    elif args.stage == 1:
        # Cache retention is (1 - local_merge_ratio) ** merging_iterations, so
        # with 10 iterations these ratios retain ~60%, ~35%, ~11% of the ~460
        # accumulated background tokens. r_match controls how much of the
        # background each matching frame reuses from that cache.
        grid = {
            "merging_iterations": [10],
            "local_merge_ratio": [0.05, 0.1, 0.2],
            "r_match": [0.5, 0.65, 0.8, 0.9],
            "bg_ratio": [0.5],
        }
    else:
        assert args.r_match and args.local_merge_ratio, "--stage 2 needs stage-1 winners"
        grid = {
            "merging_iterations": [10],
            "local_merge_ratio": [args.local_merge_ratio],
            "r_match": [args.r_match],
            "bg_ratio": [0.3, 0.4, 0.5, 0.6, 0.7],
        }

    report(sweep(grid, args.n_items, args.out))


if __name__ == "__main__":
    main()
