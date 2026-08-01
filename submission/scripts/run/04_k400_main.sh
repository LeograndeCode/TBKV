#!/usr/bin/env bash
# Table 3 (tab:k400) -- action recognition on Kinetics-400 with ViViT-B.
#
# Dataset : Kinetics-400, full 19,877-clip validation split.
# Protocol: SINGLE PASS, the mirror image of the VID replay protocol. The
#           first 4 temporal steps of each view build the persistent memory in
#           caching mode and the remaining 12 run in matching mode; BOTH are
#           counted. Nothing is amortised away.
#           Outputs report "caching A + matching B = TOTAL C" GFLOPs/clip:
#           "Whole clip" in the table is C, "Matching" is B.
# Runtime : ~4-6 h per row.
#
# Each step is skipped if its output is already present; FORCE=1 redoes it.
#
# Note that no PSM row passes a replay_matching override. Kinetics-400 uses
# single-pass accounting (caching and matching both counted); adding
# replay_matching=true would switch to the two-pass detection protocol and
# produce a different, cheaper-looking number that the paper does not report.
#
# NAMING TRAP (this bites people): the config `eventful_psm_24/48/96` defaults
# to cache_reuse=0.0, which disables the PSM filter entirely -- those runs ARE
# the Eventful baseline rows. PSM rows are the same configs with
# cache_reuse=0.95 merge_iterations=2 passed on the command line.
#
# Usage:  bash scripts/run/04_k400_main.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

banner "Table 3 (tab:k400) -- ViViT-B / Kinetics-400, full 19,877-clip split"

# --- Dense ViViT-B reference ------------------- 78.64 / 93.60, 3360 GF/clip
if skip_if_done "$K400_RES/base/output.txt" "dense ViViT-B"; then
    run python scripts/evaluate/vivit_kinetics400.py base
fi

# --- Eventful rows (cache_reuse=0.0 by config default) ---------------------
#   k=24  ->  62.38 / 82.23,  caching 618  + matching 435  = 1053
#   k=48  ->  67.54 / 87.26,  caching 1016 + matching 860  = 1877
#   k=96  ->  75.71 / 92.35,  caching 1814 + matching 1711 = 3525
for k in 24 48 96; do
    if skip_if_done "$K400_RES/eventful_psm_$k/output.txt" "Eventful k=$k"; then
        run python scripts/evaluate/eventful_psm_vivit_kinetics400.py "eventful_psm_$k"
    fi
done

# --- PSM rows (gamma=0.95, T=2 -- selected on the 10% split) ---------------
#   k=24  ->  59.82 / 79.83,  caching 618  + matching 27 = 645
#   k=48  ->  58.37 / 78.61,  caching 1016 + matching 45 = 1062
if skip_if_done "$K400_RES/eventful_psm_24-cache_reuse=0.95-merge_iterations=2/output.txt" "PSM k=24"; then
    run python scripts/evaluate/eventful_psm_vivit_kinetics400.py eventful_psm_24 \
        cache_reuse=0.95 merge_iterations=2
fi

if skip_if_done "$K400_RES/eventful_psm_48-cache_reuse=0.95-merge_iterations=2/output.txt" "PSM k=48"; then
    run python scripts/evaluate/eventful_psm_vivit_kinetics400.py eventful_psm_48 \
        cache_reuse=0.95 merge_iterations=2
fi

#   k=96  ->  58.43 / 78.49,  caching 1814 + matching 98 = 1912
if skip_if_done "$K400_RES/eventful_psm_96-cache_reuse=0.95-merge_iterations=2/output.txt" "PSM k=96"; then
    run python scripts/evaluate/eventful_psm_vivit_kinetics400.py eventful_psm_96 \
        cache_reuse=0.95 merge_iterations=2
fi

banner "Done -- Table 3 (all 7 rows)"
