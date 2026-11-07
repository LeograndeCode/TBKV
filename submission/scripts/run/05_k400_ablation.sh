#!/usr/bin/env bash
# Table 2 (tab:k400-ablation) -- reuse-fraction ablation on Kinetics-400.
#
# gamma (cache_reuse) sweep at the smallest host budget k=24, on the fixed 10%
# subsample (1,988 of 19,877 clips). T is left at the config default of 4.
#
# This is the table behind the paper's central contrast: on a global
# class-token readout, accuracy is a STEP rather than the slope seen on dense
# detection -- it drops once when the filter engages, then stays flat to within
# 0.1 Top-1 across the whole sweep while matching cost falls 12x.
#
# Expected (Top-1 % / matching GF/clip / whole-clip GF):
#   gamma=0.25  ->  58.75 / 329 / 946
#   gamma=0.50  ->  58.65 / 222 / 840
#   gamma=0.75  ->  58.65 / 116 / 734
#   gamma=0.95  ->  58.70 /  27 / 645
#
# Runtime: ~45 min per cell, ~3 h total.
#
# NOTE: the bare config `eventful_psm` is k=24 with cache_reuse=0.5 as its
# default; passing cache_reuse= on the command line overrides it, which is
# what makes this a clean sweep.
#
# Usage:  bash scripts/run/05_k400_ablation.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

banner "Table 2 (tab:k400-ablation) -- gamma sweep, 10% K400 split (1,988 clips)"

for cr in 0.25 0.5 0.75 0.95; do
    d="$K400_RES/eventful_psm-cache_reuse=$cr-n_items=1988"
    if skip_if_done "$d/output.txt" "gamma=$cr"; then
        run python scripts/evaluate/eventful_psm_vivit_kinetics400.py eventful_psm \
            cache_reuse=$cr n_items=1988
    fi
done

banner "Done -- Table 2"
cat <<'EOF'
  The paper also states that sweeping T at fixed gamma moves Top-1 by less
  than the reporting precision on this benchmark, and that T=2 is carried to
  full scale. To reproduce that claim, add merge_iterations to the loop:

      for cr in 0.25 0.5 0.75 0.95; do
        for mi in 2 4 8; do
          python scripts/evaluate/eventful_psm_vivit_kinetics400.py eventful_psm \
              cache_reuse=$cr merge_iterations=$mi n_items=1988
        done
      done
EOF
