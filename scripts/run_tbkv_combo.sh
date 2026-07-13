#!/usr/bin/env bash
# TBKV + secondary-technique fusion sweep (n_items=10).
#   ViViT/Kinetics400  : TBKV match -> substitute tokens -> secondary on fresh KV
#   ViTDet/VID         : TBKV token-skip -> secondary prunes/defers active tokens
# Secondary in {none, eventful, maskvd}; keep_ratio controls the second stage.
set -u
PY=/home/cc/miniconda3/envs/eventful-transformer/bin/python
OUT=/dev/shm/combo
mkdir -p "$OUT/logs"
cd /home/cc/TBKV

KEEP="${KEEP:-0.5}"
N="${N:-10}"

run_vivit () {
  local sec="$1" name="vivit_${1}"
  echo ">>> $name (secondary=$sec keep=$KEEP)"
  TMPDIR=/dev/shm PYTHONWARNINGS=ignore "$PY" -u scripts/evaluate/tbkv_vivit_kinetics400.py \
    tbkv_combo n_items="$N" secondary="$sec" secondary_keep="$KEEP" \
    _name="$name" "_output=$OUT/$name/" > "$OUT/logs/$name.log" 2>&1
  grep -E "Top-1 Accuracy|Top-5 Accuracy|Net FLOPs savings\b|net FLOPs savings / frame" \
    "$OUT/logs/$name.log" | sed "s/^/    /"
}

run_vitdet () {
  local sec="$1" name="vitdet_${1}"
  echo ">>> $name (secondary=$sec keep=$KEEP)"
  TMPDIR=/dev/shm PYTHONWARNINGS=ignore "$PY" -u scripts/evaluate/tbkv_vitdet_vid.py \
    tbkv_combo n_items="$N" token_skip=true cache_period=8 \
    secondary="$sec" secondary_keep="$KEEP" \
    _name="$name" "_output=$OUT/$name/" > "$OUT/logs/$name.log" 2>&1
  grep -E "amortized_gflops @T=100|map_50:" "$OUT/logs/$name.log" | sed "s/^/    /"
}

for sec in none eventful maskvd; do run_vivit  "$sec"; done
for sec in none eventful maskvd; do run_vitdet "$sec"; done
echo "COMBO SWEEP DONE"
