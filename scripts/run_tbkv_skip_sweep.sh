#!/usr/bin/env bash
# Run the TBKV token-skip configuration sweep on ImageNet VID (n_items=10, 672px).
# Each config's full stdout is saved to its own log for later parsing.
set -u
PY=/home/cc/miniconda3/envs/eventful-transformer/bin/python
OUT=/dev/shm/compare
mkdir -p "$OUT/logs"
cd /home/cc/TBKV

run () {
  local name="$1" rmatch="$2" bg="$3" period="$4"
  echo ">>> $name  (r_match=$rmatch bg_ratio=$bg cache_period=$period)"
  TMPDIR=/dev/shm PYTHONWARNINGS=ignore "$PY" -u scripts/evaluate/tbkv_vitdet_vid.py \
    _tbkv n_items=10 token_skip=true r_match="$rmatch" bg_ratio="$bg" cache_period="$period" \
    _name="$name" "_output=$OUT/$name/" > "$OUT/logs/$name.log" 2>&1
  grep -E "gflops_matching_frame|gflops_saved|map_50:" "$OUT/logs/$name.log" | sed "s/^/    /"
}

run tbkv_skip_conservative 0.3 0.4 8
run tbkv_skip_aggressive   0.7 0.6 8
run tbkv_skip_max          0.9 0.7 8
echo "SWEEP DONE"
