#!/usr/bin/env bash
# Table 1 (tab:vid), 1024x1024 block.
#
# Same protocol, same common budget k=512, same 639-video split as the 672
# block -- only the input resolution changes. Because k is held fixed in
# ABSOLUTE terms, k=512 is 29% of the 1,764 tokens at 672 but only 12.5% of the
# 4,096 tokens at 1024; that is deliberate (see the paper's "Scaling to
# 1024x1024" paragraph), not an oversight.
#
# Runtime: ~8-10 h per row (the dense row took ~10 h), ~45 h for all five.
#
# Each step is skipped if its output is already present; FORCE=1 redoes it.
# MaskVD at 1024 is the one row the paper does not report -- its command is
# given, commented out, at the bottom.
#
# Usage:  bash scripts/run/02_vid_1024.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

banner "Table 1 (tab:vid) -- ViTDet-B / ImageNet VID @ 1024x1024, full split"
gpu_busy_warning

K='token_top_k=[512]'

# --- dense ViTDet-B -------------------------------------------- 467.4 GF, 82.93
if skip_if_done "$VID_RES/base_1024/output.txt" "dense 1024"; then
    run python scripts/evaluate/vitdet_vid.py base_1024
fi

# --- Eventful ---------------------------- 87.8 GF, 145.2 ms, 8045 MB, 79.42
# The host PSM is layered on, and the reference for the "21% below Eventful"
# claim and the 8045 -> 8062 MB memory decomposition. Latency and peak memory
# print to stdout on every run of vitdet_vid.py (no flag needed).
if skip_if_done "$VID_RES/temporal_1024-token_top_k=[512]/output.txt" "Eventful 1024 k=512"; then
    run python scripts/evaluate/vitdet_vid.py temporal_1024 "$K"
fi

# --- Eventful + spatial pool ------------------------------------ 70.9 GF, 76.98
if skip_if_done "$VID_RES/spatiotemporal_1024-token_top_k=[512]/output.txt" "Eventful+pool 1024"; then
    run python scripts/evaluate/vitdet_vid.py spatiotemporal_1024 "$K"
fi

# --- STGT ------------------------------------------------------ 164.0 GF, 76.70
if skip_if_done "$VID_RES/stgt_1024-token_top_k=[512]/output.txt" "STGT 1024"; then
    run python scripts/evaluate/vitdet_vid.py stgt_1024 "$K"
fi

# --- PSM (layered) ---------------------------------------------- 69.1 GF, 75.57
PSM1024="$VID_RES/psm_eventful_filter_1024-token_top_k=[512]-cache_reuse=0.25-merge_iterations=6-warmup=4-replay_matching=true-measure_latency=true"
if skip_if_done "$PSM1024/output.txt" "PSM 1024"; then
    run python scripts/evaluate/psm_eventful_vitdet_vid.py psm_eventful_filter_1024 \
        "$K" cache_reuse=0.25 merge_iterations=6 warmup=4 \
        replay_matching=true measure_latency=true
fi

# ---------------------------------------------------------------------------
#  NOT RUN -- MaskVD at 1024, the one row the paper's 1024 block omits.
#
#  Requires the input_size override added for this submission
#  (scripts/evaluate/maskvd_vitdet_vid.py previously hardcoded 672 in three
#  places: the model input_shape, the VIDResize transform, and the
#  mask-construction grid). input_size=1024 threads all three, and the resize
#  rule matches vitdet_vid.py exactly (640*size//1024). Verified to run.
#
#      run python scripts/evaluate/maskvd_vitdet_vid.py \
#          input_size=1024 _output="$VID_RES/maskvd_1024/"
# ---------------------------------------------------------------------------

banner "Done -- 1024 block (all 5 reported rows)"
