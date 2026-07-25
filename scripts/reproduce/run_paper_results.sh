#!/bin/bash
# ============================================================================
#  TempoMem (AAAI-27) -- reproduction script for every number in the paper.
#
#  Scope: ViTDet-B / ImageNet VID at 672x672. Table 1, Figure 1, Figure 2.
#
#  Hardware used for the reported numbers: 1x NVIDIA Quadro RTX 6000 (24 GB).
#  FLOPs and mAP are deterministic and hardware-independent; latency and peak
#  memory are not, and will differ on other GPUs.
#
#  Runtime: the six full-split runs take roughly 4 h each (~24 h total); the
#  16-cell ablation grid takes roughly 1 h per cell (~16 h). Steps are
#  independent -- run them individually if preferred.
#
#  IMPORTANT: run steps SEQUENTIALLY on an otherwise idle GPU. Latency and
#  peak-memory numbers are invalid if two jobs share the device.
#
#  Usage:
#     bash scripts/reproduce/run_paper_results.sh          # everything
#     bash scripts/reproduce/run_paper_results.sh table    # Table 1 / Figure 1
#     bash scripts/reproduce/run_paper_results.sh ablation # Figure 2
#  Then verify against the reported values with:
#     python scripts/reproduce/verify_paper_results.py
# ============================================================================
set -u
source /home/cc/miniconda3/etc/profile.d/conda.sh
conda activate eventful-transformer
cd "$(dirname "$0")/../.."

WHAT="${1:-all}"
K='token_top_k=[512]'
run() { echo -e "\n### $* \n"; PYTHONUNBUFFERED=1 "$@"; }

# --------------------------------------------------------------------------
#  Table 1 / Figure 1 -- full 639-video validation split, replay-matching
#  accounting. Every row is one command.
# --------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = table ]; then

  # (1) Dense ViTDet-B baseline
  #     -> results/evaluate/vitdet_vid/base_672/
  run python scripts/evaluate/vitdet_vid.py base_672

  # (2) Eventful. NOTE: temporal_672.yml defaults to the six-k sweep
  #     [128,256,384,512,768,1024]; the paper quotes the k=512 block, and the
  #     k=384 block for the matched-compute comparison. Pin the budget with
  #     `token_top_k=[512]` if only the table row is needed (~4 h vs ~26 h).
  #     -> results/evaluate/vitdet_vid/temporal_672/
  run python scripts/evaluate/vitdet_vid.py temporal_672

  # (3) Eventful + spatial pooling (its spatio-temporal variant)
  #     -> results/evaluate/vitdet_vid/spatiotemporal_672-token_top_k=[512]/
  run python scripts/evaluate/vitdet_vid.py spatiotemporal_672 "$K"

  # (4) STGT
  #     -> results/evaluate/vitdet_vid/stgt_672-token_top_k=[512]/
  run python scripts/evaluate/vitdet_vid.py stgt_672 "$K"

  # (5) MaskVD. Defaults to /dev/shm (volatile); pin an output dir.
  #     -> results/evaluate/vitdet_vid/maskvd_672/
  run python scripts/evaluate/maskvd_vitdet_vid.py \
      _output=results/evaluate/vitdet_vid/maskvd_672/

  # (6) TempoMem, layered. gamma=0.25, T=6 -- the best cell of the Figure-2
  #     grid. measure_latency=true is REQUIRED: unlike vitdet_vid.py, this
  #     script gates the latency/memory harness behind that flag.
  #     -> results/evaluate/vitdet_vid/tbkv_eventful_filter_672-token_top_k=[512]
  #        -cache_reuse=0.25-merge_iterations=6-warmup=4-replay_matching=true
  #        -measure_latency=true/
  run python scripts/evaluate/tbkv_eventful_vitdet_vid.py tbkv_eventful_filter_672 \
      "$K" cache_reuse=0.25 merge_iterations=6 warmup=4 \
      replay_matching=true measure_latency=true
fi

# --------------------------------------------------------------------------
#  Figure 2 -- reuse-fraction ablation on the fixed 10% split (64 videos).
#  The paper plots the best T per gamma; the full 4x4 grid is run so that
#  claim is checkable.
# --------------------------------------------------------------------------
if [ "$WHAT" = all ] || [ "$WHAT" = ablation ]; then
  for cr in 0.25 0.5 0.75 0.95; do
    for mi in 2 4 6 8; do
      # -> ...-warmup=4-cache_reuse=$cr-merge_iterations=$mi-n_items=64-replay_matching=true/
      run python scripts/evaluate/tbkv_eventful_vitdet_vid.py tbkv_eventful_filter_672 \
          "$K" warmup=4 cache_reuse=$cr merge_iterations=$mi \
          n_items=64 replay_matching=true
    done
  done
fi

echo -e "\nDone. Verify with:  python scripts/reproduce/verify_paper_results.py"
