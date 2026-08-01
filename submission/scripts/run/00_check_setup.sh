#!/usr/bin/env bash
# Pre-flight check. Runs nothing expensive: verifies the environment, the
# model weights, the datasets and the GPU before you commit to ~125 GPU-hours.
#
# Usage:  bash scripts/run/00_check_setup.sh
# Exit 0 if everything needed for the full reproduction is present.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

fail=0
ok()   { printf '  [ ok ] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=1; }
warn() { printf '  [warn] %s\n' "$*"; }

banner "1. Python environment"
python - <<'PY'
import importlib, sys

# Required by at least one reported table or figure.
need = {
    "torch": "2.0", "torchvision": "0.15", "torchmetrics": "0.11",
    "numpy": "1.23", "matplotlib": "3.7", "omegaconf": "2.3",
    "einops": "0.8", "detectron2": "0.6", "tqdm": None,
}
# Declared in environment.yml but not imported by any code path behind a
# reported number. Reported for information only, never as a failure.
optional = {
    "av": "declared for video decoding; unused by the reproduction path",
    "cv2": "declared for the Detectron visualizer; unused by the reproduction path",
}
bad = []
for mod, want in need.items():
    try:
        m = importlib.import_module(mod)
        got = getattr(m, "__version__", "?")
        flag = "" if (want is None or str(got).startswith(want)) else f"  (expected {want}.x)"
        print(f"  [ ok ] {mod:<12} {got}{flag}")
    except Exception as exc:
        print(f"  [FAIL] {mod:<12} {type(exc).__name__}: {exc}")
        bad.append(mod)
for mod, why in optional.items():
    try:
        m = importlib.import_module(mod)
        print(f"  [ ok ] {mod:<12} {getattr(m, '__version__', '?')}  (optional)")
    except Exception:
        print(f"  [ -- ] {mod:<12} not installed (optional: {why})")
print(f"  torch.cuda.is_available(): {__import__('torch').cuda.is_available()}")
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && fail=1

banner "2. Model weights  (weights/)"
# All five files ship in weights/ as regular files. -s and du -L also accept
# symlinks, so a checkout that links them elsewhere still passes.
for w in vitdet_b_vid.pth \
         vivit_b_kinetics400.pth \
         vivit_b_kinetics400_final_24.pth \
         vivit_b_kinetics400_final_48.pth \
         vivit_b_kinetics400_final_96.pth; do
    if [ -s "weights/$w" ]; then
        note=""
        if [ -L "weights/$w" ]; then
            note="  -> $(readlink "weights/$w")"
        fi
        ok "weights/$w  ($(du -hL "weights/$w" | cut -f1))$note"
    elif [ -L "weights/$w" ]; then
        bad "weights/$w  BROKEN SYMLINK -> $(readlink "weights/$w")"
    else
        bad "weights/$w  MISSING"
    fi
done

banner "3. Datasets  (data/ ships empty; see README section 2)"
if [ -s data/vid/data.tar ] || [ -d data/vid/unpacked ]; then
    ok "data/vid  ($(du -sh data/vid 2>/dev/null | cut -f1))"
else
    bad "data/vid/data.tar  MISSING  -- place the official ILSVRC2015 VID archive there; needed for every ViTDet/VID table and figure"
fi
if [ -n "$(ls -A data/kinetics400 2>/dev/null)" ]; then
    ok "data/kinetics400  ($(du -sh data/kinetics400 2>/dev/null | cut -f1))"
else
    warn "data/kinetics400 is empty -- the loader downloads it automatically on first use (~123 GB)"
fi

banner "4. Third-party baseline code (optional)"
# The Eventful host and the STGT baseline are implemented inside src/core/, so
# nothing external is needed for them. Only the MaskVD row of Table 1 depends on
# a third-party implementation, which is cited in the paper but not
# redistributed here. Its absence is NOT a failure.
if [ -d "MaskVD" ] && [ -n "$(ls -A MaskVD 2>/dev/null)" ]; then
    ok "MaskVD/ present -- the MaskVD row of Table 1 can be reproduced"
else
    warn "MaskVD/ not present -- every row except MaskVD in Table 1 still runs"
fi

banner "5. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader \
        | sed 's/^/  /'
    gpu_busy_warning
else
    warn "nvidia-smi not found; CPU-only runs are far too slow for these experiments"
fi

banner "Result"
if [ "$fail" -eq 0 ]; then
    echo "  Setup looks complete. Next:  bash scripts/run/01_vid_672.sh"
    exit 0
fi
echo "  One or more required items are missing (see [FAIL] above)."
echo "  See README.md section 2 (Datasets) for how to obtain them."
exit 1
