#!/usr/bin/env python3
"""Sweep cache_reuse x merge_iterations for EventfulTBKV on ViViT and ViTDet.

cr=0 is the merge-invariant baseline (cache unused = exact Eventful), run once.
For cr>0 we cross cache_reuse with merge_iterations. Results append to CSV so a
killed sweep keeps what it measured.
"""
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FLOP_RE = re.compile(r"(\w+_flops):\s*([\d.e+-]+)")
MAP50_RE = re.compile(r"map_50:\s*([\d.]+)")
TOP1_RE = re.compile(r"Top-1 Accuracy\s*:\s*([\d.]+)")
TOP5_RE = re.compile(r"Top-5 Accuracy\s*:\s*([\d.]+)")
VIVIT_MATCH_RE = re.compile(r"matching (\d+)")


def run(cmd, timeout=3600):
    t0 = time.time()
    p = subprocess.run([sys.executable] + cmd, cwd=REPO, capture_output=True,
                       text=True, timeout=timeout)
    return p.stdout + p.stderr, round(time.time() - t0)


def sweep_vivit(out_csv, n_items=100):
    grid = [(0.0, 4)] + [(cr, mi) for cr in (0.25, 0.5, 0.75) for mi in (2, 4, 8)]
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cache_reuse", "merge_iterations", "top1", "top5",
                    "matching_gflops_clip", "matching_gflops_frame", "seconds"])
        for cr, mi in grid:
            out, secs = run([
                "scripts/evaluate/eventful_tbkv_vivit_kinetics400.py", "eventful_tbkv",
                f"n_items={n_items}", f"cache_reuse={cr}", f"merge_iterations={mi}",
                f"_name=sw_v_cr{cr}_mi{mi}",
            ])
            t1 = TOP1_RE.search(out); t5 = TOP5_RE.search(out)
            mm = VIVIT_MATCH_RE.search(out)
            if not t1:
                print(f"[vivit cr={cr} mi={mi}] FAILED\n" + "\n".join(out.splitlines()[-4:]), flush=True)
                continue
            gclip = float(mm.group(1)) if mm else float("nan")
            row = [cr, mi, float(t1.group(1)), float(t5.group(1)),
                   gclip, round(gclip / 16, 2), secs]
            w.writerow(row); fh.flush()
            print(f"[vivit cr={cr} mi={mi}] top1={row[2]} top5={row[3]} "
                  f"match={gclip:.0f} GF/clip ({gclip/16:.1f}/frame) ({secs}s)", flush=True)


def sweep_vitdet(out_csv, n_items=25, k=512):
    grid = [(0.0, 4)] + [(cr, mi) for cr in (0.25, 0.5, 0.75) for mi in (2, 4, 8)]
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cache_reuse", "merge_iterations", "map50",
                    "gflops_frame", "seconds"])
        for cr, mi in grid:
            out, secs = run([
                "scripts/evaluate/tbkv_eventful_vitdet_vid.py", "tbkv_eventful_filter_672",
                f"n_items={n_items}", "warmup=4", f"cache_reuse={cr}",
                f"merge_iterations={mi}", f"token_top_k=[{k}]",
                f"_name=sw_d_cr{cr}_mi{mi}",
            ], timeout=5400)
            m50 = MAP50_RE.search(out)
            if not m50:
                print(f"[vitdet cr={cr} mi={mi}] FAILED\n" + "\n".join(out.splitlines()[-4:]), flush=True)
                continue
            gf = sum(float(v) for _, v in FLOP_RE.findall(out)) / 1e9
            row = [cr, mi, float(m50.group(1)), round(gf, 2), secs]
            w.writerow(row); fh.flush()
            print(f"[vitdet cr={cr} mi={mi}] mAP50={row[2]:.3f} "
                  f"{gf:.2f} GF/frame ({secs}s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    outdir = REPO / "results" / "sweeps"
    outdir.mkdir(parents=True, exist_ok=True)
    if which in ("vivit", "both"):
        print("=== ViViT cr x merge_iterations sweep ===", flush=True)
        sweep_vivit(outdir / "cr_merge_vivit.csv")
    if which in ("vitdet", "both"):
        print("=== ViTDet cr x merge_iterations sweep ===", flush=True)
        sweep_vitdet(outdir / "cr_merge_vitdet.csv")
    print("SWEEP-DONE", flush=True)
