#!/usr/bin/env bash
# Figure 2 (fig:ablation) -- reuse-fraction ablation on ImageNet VID.
#
# Full 4x4 grid: gamma (cache_reuse) x T (merge_iterations), on the fixed 10%
# subsample (64 of 639 videos) that every ablation in the paper shares. The
# figure plots the best T per gamma; the whole grid is run so that claim is
# checkable, and so the T-insensitivity result has evidence behind it.
#
# The mAP deltas and compute savings quoted for this figure are relative to
# the full-split dense reference (base_672: 82.28 mAP@50, 174.5 GF/frame),
# i.e. the Table 1 dense row -- run 01_vid_672.sh first if it is missing.
#
# Runtime: ~1 h per cell, ~16 h for the grid.
#
# Selected operating point carried into Table 1: gamma=0.25, T=6.
#
# Usage:  bash scripts/run/03_vid_ablation.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

banner "Figure 2 (fig:ablation) -- gamma x T grid, 10% VID split (64 videos)"

K='token_top_k=[512]'

# --- The 4x4 grid -----------------------------------------------------------
# Expected best-T mAP@50 per gamma (single run, ~1 point of noise):
#   gamma=0.25  T=6  ->  79.49 mAP@50,  46.6 GF/frame
#   gamma=0.50  T=6  ->  75.02 mAP@50,  32.9 GF/frame
#   gamma=0.75  T=8  ->  63.63 mAP@50,  19.3 GF/frame
#   gamma=0.95  T=2  ->  38.26 mAP@50,   8.4 GF/frame
for cr in 0.25 0.5 0.75 0.95; do
    for mi in 2 4 6 8; do
        d="$VID_RES/psm_eventful_filter_672-token_top_k=[512]-warmup=4-cache_reuse=$cr-merge_iterations=$mi-n_items=64-replay_matching=true"
        if skip_if_done "$d/output.txt" "gamma=$cr T=$mi"; then
            run python scripts/evaluate/psm_eventful_vitdet_vid.py psm_eventful_filter_672 \
                "$K" warmup=4 cache_reuse=$cr merge_iterations=$mi \
                n_items=64 replay_matching=true
        fi
    done
done

banner "Done -- ablation grid"
cat <<'EOF'
  NOTE ON DIRECTORY NAMES: overrides appear in the output directory name in
  the ORDER THEY ARE PASSED on the command line. The names above assume the
  argument order used here (warmup, cache_reuse, merge_iterations, n_items,
  replay_matching). Reordering the arguments produces a differently-named
  directory that the figure script will not find.

  Verify with:  python scripts/reproduce/verify_paper_results.py
EOF
