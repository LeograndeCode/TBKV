#!/usr/bin/env bash
# OCTA hyperparameter sweep on Kinetics-400 (ViViT-B).
# Same eval harness / settings as the stored SOTA baselines:
#   n_items=10, frame_split=16 (default)  ->  directly comparable GFLOPs.
# Each config's full stdout is saved to its own log + output dir for parsing.
set -u
PY=/home/cc/miniconda3/envs/eventful-transformer/bin/python
OUT=results/evaluate/vivit_kinetics400/octa_sweep
N=${N:-10}
FS=${FS:-16}
mkdir -p "$OUT/logs"
cd /home/cc/TBKV

run () {
  local name="$1"; shift
  echo ">>> $name  ($*)"
  TMPDIR=/dev/shm PYTHONWARNINGS=ignore "$PY" -u scripts/evaluate/octa_vivit_kinetics400.py \
    octa n_items="$N" frame_split="$FS" "$@" \
    _name="octa_sweep/$name" > "$OUT/logs/$name.log" 2>&1
  grep -E "Top-1 Accuracy|Top-5 Accuracy|Total GFLOPs \(MATCHING|avg matching frames" \
    "$OUT/$name/output.txt" 2>/dev/null | sed "s/^/    /"
}

# ---- center / default -------------------------------------------------------
run default cluster_similarity_threshold=0.6 merging_iterations=2 object_match_threshold=0.55 offset_window_size=2.0

# ---- A. clustering aggressiveness (cluster_similarity_threshold) ------------
run thr050 cluster_similarity_threshold=0.50 merging_iterations=2 object_match_threshold=0.55 offset_window_size=2.0
run thr070 cluster_similarity_threshold=0.70 merging_iterations=2 object_match_threshold=0.55 offset_window_size=2.0
run thr080 cluster_similarity_threshold=0.80 merging_iterations=2 object_match_threshold=0.55 offset_window_size=2.0

# ---- B. merging_iterations --------------------------------------------------
run iters1 cluster_similarity_threshold=0.6 merging_iterations=1 object_match_threshold=0.55 offset_window_size=2.0
run iters3 cluster_similarity_threshold=0.6 merging_iterations=3 object_match_threshold=0.55 offset_window_size=2.0

# ---- C. cache-reuse strictness (object_match_threshold) --------------------
run match045 cluster_similarity_threshold=0.6 merging_iterations=2 object_match_threshold=0.45 offset_window_size=2.0
run match065 cluster_similarity_threshold=0.6 merging_iterations=2 object_match_threshold=0.65 offset_window_size=2.0

# ---- D. local matching window (offset_window_size) -------------------------
run win30 cluster_similarity_threshold=0.6 merging_iterations=2 object_match_threshold=0.55 offset_window_size=3.0

echo "SWEEP DONE"
