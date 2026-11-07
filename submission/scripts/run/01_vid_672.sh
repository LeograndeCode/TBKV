#!/usr/bin/env bash
# Table 1 (tab:vid), 672x672 block -- all six rows -- and the data behind
# Figure 1 (fig:frontier).
#
# Dataset : ImageNet VID, full 639-video validation split.
# Protocol: replay matching for PSM (4 warm-up frames, discarded and uncounted;
#           the scored pass replays every video from frame 0). Baselines run
#           the same frames under the same accounting.
# Runtime : ~4 h per row on a Quadro RTX 6000, ~24 h total.
#
# RUN THIS ON AN IDLE GPU. mAP and GFLOPs are deterministic, but the latency
# and peak-memory columns are meaningless if another job shares the device.
#
# Usage:  bash scripts/run/01_vid_672.sh
#         FORCE=1 bash scripts/run/01_vid_672.sh    # redo finished rows

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

banner "Table 1 (tab:vid) -- ViTDet-B / ImageNet VID @ 672x672, full split"
gpu_busy_warning

K='token_top_k=[512]'

# --- Row 1: dense ViTDet-B ------------------------------------- 174.5 GF, 82.28
if skip_if_done "$VID_RES/base_672/output.txt" "dense 672"; then
    run python scripts/evaluate/vitdet_vid.py base_672
fi

# --- Row 2: Eventful ------------------------------------------- 60.7 GF, 81.80
# temporal_672.yml defaults to a six-budget sweep [128..1024] (~26 h). The
# table needs only k=512, so the budget is pinned here (~4 h).
if skip_if_done "$VID_RES/temporal_672-token_top_k=[512]/output.txt" "Eventful 672 k=512"; then
    run python scripts/evaluate/vitdet_vid.py temporal_672 "$K"
fi

# --- Row 3: Eventful + spatial pool ---------------------------- 52.8 GF, 79.48
# "spatiotemporal" = Eventful gating compounded with 2x2 key/value pooling.
if skip_if_done "$VID_RES/spatiotemporal_672-token_top_k=[512]/output.txt" "Eventful+pool 672"; then
    run python scripts/evaluate/vitdet_vid.py spatiotemporal_672 "$K"
fi

# --- Row 4: STGT ----------------------------------------------- 68.5 GF, 80.45
if skip_if_done "$VID_RES/stgt_672-token_top_k=[512]/output.txt" "STGT 672"; then
    run python scripts/evaluate/vitdet_vid.py stgt_672 "$K"
fi

# --- Row 5: MaskVD --------------------------------------------- 80.9 GF, 82.05
# This script defaults its output to /dev/shm (volatile) -- pin it explicitly.
if skip_if_done "$VID_RES/maskvd_672/results.txt" "MaskVD 672"; then
    run python scripts/evaluate/maskvd_vitdet_vid.py \
        _output="$VID_RES/maskvd_672/"
fi

# --- Row 6: PSM (layered) -------------------------------------- 46.6 GF, 79.87
# gamma=0.25, T=6: the operating point selected on the 10% split (03_vid_ablation.sh).
# measure_latency=true is REQUIRED -- unlike vitdet_vid.py, this script gates
# the latency/peak-memory harness behind that flag and silently omits the
# columns without it.
PSM672="$VID_RES/psm_eventful_filter_672-token_top_k=[512]-cache_reuse=0.25-merge_iterations=6-warmup=4-replay_matching=true-measure_latency=true"
if skip_if_done "$PSM672/output.txt" "PSM 672"; then
    run python scripts/evaluate/psm_eventful_vitdet_vid.py psm_eventful_filter_672 \
        "$K" cache_reuse=0.25 merge_iterations=6 warmup=4 \
        replay_matching=true measure_latency=true
fi

banner "Done -- 672 block"
cat <<'EOF'
  Latency and peak memory are printed to STDOUT by each run as
      Latency: <ms> ms
      Memory:  <MB> MB
  They are NOT written into the result directories. Capture this script's
  output (e.g. `bash ... 2>&1 | tee vid672.log`) or read them back from the
  scrollback when filling the Lat./Mem. columns of Table 1.

  Verify the accuracy and GFLOPs columns with:
      python scripts/reproduce/verify_paper_results.py
EOF
