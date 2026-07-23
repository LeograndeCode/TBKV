#!/bin/bash
# Re-measure Eventful (temporal_672) at k=512 on full ImageNet VID, capturing
# latency and peak memory reliably.
#
# WHY THIS EXISTS
#   scripts/evaluate/vitdet_vid.py already measures latency/memory
#   unconditionally (CUDA events + max_memory_allocated, see
#   evaluate_vitdet_metrics). But it emits them with a plain print() to
#   STDOUT, whereas the mAP/FLOPs block goes through tee_print(), which also
#   writes output.txt. In the previous six-k run the shared append-log lost
#   most of stdout, so only 2 of 6 latency pairs survived while output.txt
#   kept all 6 metric blocks.
#
#   Fixes here:
#     1. stdout gets its OWN file (never the shared queue log)
#     2. stderr (tqdm progress) is split into a separate file so the
#        carriage returns cannot interleave with the numbers we care about
#     3. PYTHONUNBUFFERED=1 so nothing is lost in a block buffer
#     4. token_top_k=[512] only -> ~4 h instead of ~26 h for the six-k sweep
#
# USAGE
#   bash scripts/run_temporal672_latency.sh            # wait for GPU, then run
#   NOWAIT=1 bash scripts/run_temporal672_latency.sh   # start immediately
set -u

source /home/cc/miniconda3/etc/profile.d/conda.sh
conda activate eventful-transformer
cd /home/cc/TBKV

STAMP=$(date +%Y%m%d-%H%M%S)
RESDIR="results/evaluate/vitdet_vid/temporal_672-token_top_k=[512]"
LOGDIR="results/comparison"
OUT="$LOGDIR/temporal672_k512_latency-$STAMP.out"       # stdout: the numbers
PROG="$LOGDIR/temporal672_k512_latency-$STAMP.progress" # stderr: tqdm bar
mkdir -p "$LOGDIR"

# Latency is only meaningful with EXCLUSIVE GPU access, so wait for both the
# queue controller and any in-flight eval to exit -- waiting on the running
# job alone would race the queue into launching its next step alongside us.
if [ "${NOWAIT:-0}" != "1" ]; then
  while pgrep -f 'scratchpad/run_.*\.sh' > /dev/null \
     || pgrep -f 'scripts/evaluate/.*\.py' > /dev/null; do sleep 120; done
  sleep 30   # settle: let GPU memory be released before the timing warm-up
fi

echo "start $(date -Is)" | tee "$OUT"
PYTHONUNBUFFERED=1 python scripts/evaluate/vitdet_vid.py temporal_672 \
    'token_top_k=[512]' >> "$OUT" 2> "$PROG"
rc=$?
echo "exit $rc  $(date -Is)" >> "$OUT"

echo
echo "================ Eventful temporal_672, k=512, full VID ================"
grep -E '^(Latency|Memory):' "$OUT" || echo "  !! no latency/memory captured"
if [ -f "$RESDIR/output.txt" ]; then
  grep -E 'map_50:' "$RESDIR/output.txt" | tail -1
  python - "$RESDIR/output.txt" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
fl = re.findall(r"^\s*\w+_flops:\s*([\d.eE+]+)", t, re.M)
if fl:
    print(f"    GFLOPs/frame:      {sum(map(float, fl))/1e9:.2f}")
PY
fi
echo "stdout : $OUT"
echo "progress: $PROG"
echo "results : $RESDIR/"
echo "======================================================================="
