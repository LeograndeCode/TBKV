#!/usr/bin/env python3
"""Temporal-adjacency stress test on ImageNet-VID / ViTDet.

Hypothesis under test
---------------------
Eventful's savings come from small frame-to-frame *deltas*; MaskVD's come from
the previous frame's *boxes* still being valid. Both assume temporal adjacency.
TBKV matches tokens by *content* against a persistent cache, so it is
position-agnostic and should not care how far apart consecutive inputs are.

Increasing frame_stride keeps every k-th frame: inter-frame displacement grows
while the underlying content is unchanged. If the hypothesis holds, Eventful and
MaskVD should lose accuracy faster than TBKV as k grows.

Why dense is re-run at every stride
-----------------------------------
Changing the stride changes *which frames are evaluated*, so absolute mAP moves
for every method, including the unmodified dense model. Comparing raw mAP across
strides would therefore confound "the method degraded" with "the eval set
changed". The only meaningful quantity is each method's **gap to dense at the
same stride**, so dense is measured at every stride and reported alongside.

Usage:
    python scripts/sweep_motion.py --strides 1 4 8 16 --n-items 10
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MAP_RE = re.compile(r"^\s*map:\s*([\d.]+)", re.M)
MAP50_RE = re.compile(r"^\s*map_50:\s*([\d.]+)", re.M)
# MaskVD prints its own summary rather than using the shared counting harness.
MASKVD_MAP_RE = re.compile(r"mAP:\s*([\d.]+)")
MASKVD_MAP50_RE = re.compile(r"mAP@50:\s*([\d.]+)")
MASKVD_GFLOPS_RE = re.compile(r"GFLOPs/frame:\s*([\d.]+)")
FLOP_RE = re.compile(r"(\w+_flops):\s*([\d.e+-]+)")


def _gflops_from_counts(out):
    """Sum every *_flops counter and convert to GFLOPs/frame."""
    total = sum(float(v) for _, v in FLOP_RE.findall(out))
    return total / 1e9 if total else float("nan")


METHODS = {
    "dense":    ["scripts/evaluate/vitdet_vid.py", "base_672"],
    "eventful": ["scripts/evaluate/vitdet_vid.py", "temporal_672"],
    "tbkv":     ["scripts/evaluate/tbkv_vitdet_vid.py", "tbkv_kvreuse"],
    "maskvd":   ["scripts/evaluate/maskvd_vitdet_vid.py"],
    # Eventful with TBKV rebuilt as motion compensation: identical config to
    # "eventful" except the block class, so any delta is attributable to the
    # content-addressed reference alone.
    "tbkv_eventful": ["scripts/evaluate/vitdet_vid.py", "tbkv_eventful_672"],
}


def run(method, stride, n_items, timeout=5400):
    cmd = [sys.executable] + METHODS[method] + [
        f"n_items={n_items}", f"frame_stride={stride}",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", flush=True)
        return None
    out = proc.stdout + proc.stderr

    if method == "maskvd":
        m, m50 = MASKVD_MAP_RE.search(out), MASKVD_MAP50_RE.search(out)
        g = MASKVD_GFLOPS_RE.search(out)
        gflops = float(g.group(1)) if g else float("nan")
    else:
        m, m50 = MAP_RE.search(out), MAP50_RE.search(out)
        gflops = _gflops_from_counts(out)

    if not m50:
        tail = "\n".join(out.strip().splitlines()[-5:])
        print(f"    FAILED (rc={proc.returncode})\n{tail}", flush=True)
        return None

    return {
        "map": float(m.group(1)) if m else float("nan"),
        "map_50": float(m50.group(1)),
        "gflops": gflops,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--n-items", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=["dense", "eventful", "maskvd", "tbkv"])
    ap.add_argument("--out", type=Path,
                    default=REPO / "results" / "sweeps" / "motion_stress.csv")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stride", "method", "map", "map_50", "gflops", "seconds"]
    new = not args.out.exists()
    rows = []

    with args.out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for stride in args.strides:
            for method in args.methods:
                print(f"[stride={stride:>2}] {method}", flush=True)
                res = run(method, stride, args.n_items)
                if res is None:
                    continue
                row = {"stride": stride, "method": method, **res}
                w.writerow(row)
                fh.flush()
                rows.append(row)
                print(f"    mAP50={res['map_50']:.3f}  mAP={res['map']:.3f}  "
                      f"GFLOPs={res['gflops']:.1f}  ({res['seconds']:.0f}s)",
                      flush=True)

    # Report each method's gap to dense at the SAME stride -- the only
    # stride-comparable quantity.
    print("\n" + "=" * 66)
    print("mAP@50 GAP TO DENSE (same stride). Less negative = more robust.")
    print("=" * 66)
    print(f"{'stride':>6}  " + "  ".join(f"{m:>10}" for m in args.methods if m != "dense"))
    for stride in args.strides:
        dense = next((r for r in rows if r["stride"] == stride and r["method"] == "dense"), None)
        if not dense:
            continue
        cells = []
        for m in args.methods:
            if m == "dense":
                continue
            r = next((r for r in rows if r["stride"] == stride and r["method"] == m), None)
            cells.append(f"{r['map_50'] - dense['map_50']:+10.3f}" if r else f"{'-':>10}")
        print(f"{stride:>6}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
