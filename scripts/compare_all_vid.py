#!/usr/bin/env python3
"""
Comparison runner: evaluate all methods on ImageNet VID with n_items=10.
Runs sequentially to avoid GPU contention. Results go to /dev/shm/compare/.

Usage:
    cd /home/cc/TBKV
    TMPDIR=/dev/shm python scripts/compare_all_vid.py [n_items=10]
"""
import os
import sys
import subprocess
from pathlib import Path

TBKV_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/cc/miniconda3/envs/eventful-transformer/bin/python"

def run(script_args, env_override=None):
    env = {**os.environ, "TMPDIR": "/dev/shm"}
    if env_override:
        env.update(env_override)
    cmd = [PYTHON] + script_args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(str(c) for c in script_args)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(TBKV_ROOT), env=env)
    return result.returncode


def main():
    # Parse n_items from CLI
    n_items = 10
    for arg in sys.argv[1:]:
        if arg.startswith("n_items="):
            n_items = int(arg.split("=")[1])

    out_base = f"/dev/shm/compare/n{n_items}"

    eval_script = "scripts/evaluate/vitdet_vid.py"
    maskvd_script = "scripts/evaluate/maskvd_vitdet_vid.py"
    tbkv_script = "scripts/evaluate/tbkv_vitdet_vid.py"

    print(f"\nComparing all methods on VID with n_items={n_items}")
    print(f"Results will be in {out_base}/")
    print("Methods: Baseline, STGT, Eventful, MaskVD, TBKV")

    runs = [
        # (description, script, config_name, extra_args)
        ("Baseline ViTDet-B", eval_script,
         "base_672",
         [f"n_items={n_items}", f"_output={out_base}/baseline_672/"]),

        ("STGT (EventfulTokenwiseBlock)", eval_script,
         "stgt_672",
         [f"n_items={n_items}", f"_output={out_base}/stgt_672/"]),

        ("Eventful-transformer (temporal)", eval_script,
         "temporal_672",
         [f"n_items={n_items}", f"_output={out_base}/temporal_672/"]),

        ("Eventful-transformer (spatiotemporal)", eval_script,
         "spatiotemporal_672",
         [f"n_items={n_items}", f"_output={out_base}/spatiotemporal_672/"]),
    ]

    # Run MaskVD separately (different script)
    maskvd_run = ("MaskVD", maskvd_script, None,
                  [f"n_items={n_items}", f"_output={out_base}/maskvd_672/"])

    # Run TBKV separately (different script)
    tbkv_run = ("TBKV (ours)", tbkv_script, None,
                [f"n_items={n_items}", f"_output={out_base}/tbkv_672/",
                 "r_match=0.5"])

    results = {}

    # Standard eval script runs
    for desc, script, config, extra in runs:
        args = [script]
        if config:
            args.append(config)
        args += extra
        rc = run(args)
        results[desc] = {"ok": rc == 0, "output": f"{out_base}/{extra[-1].split('/')[-2]}/"}

    # MaskVD
    desc, script, _, extra = maskvd_run
    rc = run([script] + extra)
    results[desc] = {"ok": rc == 0, "output": f"{out_base}/maskvd_672/"}

    # TBKV
    desc, script, _, extra = tbkv_run
    rc = run([script] + extra)
    results[desc] = {"ok": rc == 0, "output": f"{out_base}/tbkv_672/"}

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    for method, info in results.items():
        status = "OK" if info["ok"] else "FAILED"
        print(f"  {status:6s}  {method}: {info['output']}")

    print(f"\nResults at: {out_base}/")
    print("To view output.txt for each method:")
    for method, info in results.items():
        out_file = Path(info["output"]) / "output.txt"
        print(f"  cat {out_file}   # {method}")


if __name__ == "__main__":
    main()
