# TBKV: Token-Based KV Caching for Efficient Video Transformers

**Repository:** https://github.com/wision-lab/TBKV

## Introduction

TBKV (Token-Based KV Caching) is a method for efficient video transformer inference that leverages temporal redundancy across frames. Rather than recomputing Keys and Values (KV) from scratch at every frame, TBKV builds a compressed KV cache from a set of caching frames and reuses it during subsequent matching frames.

The core insight is that in video, background regions remain largely static across frames. TBKV exploits this by:

1. **Separating foreground and background tokens** using the attention map from the previous frame as a saliency signal. Tokens attending to high-attention regions are classified as foreground; the rest as background.
2. **Merging background tokens** into compact representations (using a ratio $r_{\text{merge}}$) before storing them in the cache. This reduces cache memory and matching cost.
3. **Matching incoming background tokens** against the cache at inference time using cosine similarity. If a token is sufficiently similar to a cached entry (controlled by $r_{\text{match}}$), its KV computation is skipped and the cached KV is reused instead.
4. **Forwarding only unmatched tokens** through the full QKV projection, then assembling the full attention input from foreground tokens + unmatched background tokens + retrieved cache entries.

This two-pass evaluation strategy — a **caching pass** over the first $P$ frames, followed by a **matching pass** over the remaining frames — achieves significant FLOPs savings with minimal accuracy degradation.

### Key Components

- **Foreground/background separation**: driven by the previous frame's attention map, without learned gates or auxiliary networks.
- **Token merging in the cache**: background tokens are merged with a local merge ratio ($r_{\text{merge}}$) so each cache entry represents multiple original tokens, reducing cache footprint.
- **Cosine similarity matching**: incoming tokens are matched to cache entries; matched tokens skip KV projection entirely.
- **Merged mode vs. raw mode**: in merged mode the cache is compressed; in raw mode all tokens are stored without merging (higher memory, higher recall).

### Efficiency Profile (ViViT-B, Kinetics-400)

| | Vanilla ViViT | TBKV (matching pass) |
|---|---|---|
| `linear_flops` | 3.218e+12 | ~2.28e+12 (**−29%**) |
| `matmul_flops` | 1.374e+11 | ~7.97e+10 (**−42%**) |
| Net KV savings/video | — | ~7.15e+09 |
| Merge factor (bg→cache) | — | ~2× per block |

## Quick Start

### Prerequisites

- Linux system (Windows users should use WSL2)
- ~50 GB free disk space (for Kinetics-400 and results)
- NVIDIA GPU with CUDA support recommended (compute capability 7.0+)

### Environment Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/wision-lab/TBKV.git
cd TBKV
```

#### 2. Install Miniconda

If you don't have Miniconda/Anaconda installed:

```bash
# Linux (x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
```

#### 3. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate eventful-transformer
```

The `environment.yml` includes Python 3.10, PyTorch 2.0 with CUDA 11.8, Detectron2 (built from source), OpenCV, FFmpeg, TensorBoard, and all required dependencies.

#### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### 5. Set PYTHONPATH

Scripts must be run from the repo root with the current directory on the Python path:

```bash
export PYTHONPATH="$PYTHONPATH:."
```

## Repository Structure

```
TBKV/
├── configs/
│   ├── evaluate/          # Evaluation configurations
│   │   ├── vivit_kinetics400/
│   │   ├── vivit_epic_kitchens/
│   │   └── vitdet_vid/
│   ├── models/            # Model architecture configs
│   ├── spatial/           # Spatial pre-caching configs
│   └── train/             # Fine-tuning configs
├── scripts/
│   ├── evaluate/          # Evaluation entry points
│   │   ├── tbkv_vivit_kinetics400.py   # TBKV evaluation
│   │   ├── vivit_kinetics400.py        # Vanilla baseline
│   │   ├── vivit_epic_kitchens.py
│   │   └── vitdet_vid.py
│   ├── convert/           # Weight conversion scripts
│   └── spatial/           # Spatial feature caching
├── src/
│   ├── core/              # Base transformer modules
│   │   ├── base.py        # ExtendedModule, Counts, FLOPs tracking
│   │   ├── blocks.py      # Standard transformer block
│   │   ├── backbones.py   # ViT backbone
│   │   └── counting.py    # CountedLinear, CountedMatmul, etc.
│   ├── tbkv/              # TBKV-specific modules
│   │   ├── tbkv_blocks.py     # TBKVBlock — caching and matching logic
│   │   ├── tbkv_backbone.py   # TBKVViTBackbone
│   │   ├── cache.py           # Cache class and cosine matching
│   │   ├── match.py           # perform_tbkv_matching
│   │   ├── merge.py           # Token merging (ToMe integration)
│   │   └── tbkv_utils.py      # extract_bg_fg_tokens, compute_merge
│   ├── models/
│   │   ├── vivit.py           # FactorizedViViT (vanilla)
│   │   ├── tbkv_vivit.py      # TBKVFactorizedViViT
│   │   └── vitdet.py          # ViTDet object detector
│   ├── datasets/
│   │   ├── kinetics400.py
│   │   ├── epic_kitchens.py
│   │   └── vid.py
│   └── utils/
│       ├── evaluate.py        # Vanilla evaluation loop
│       ├── evaluate_tbkv.py   # TBKV evaluation loop (legacy)
│       ├── config.py          # OmegaConf config loading
│       └── misc.py            # TopKAccuracy, tee_print, etc.
├── utils/
│   └── test_evaluate.py   # New evaluation framework with detailed stats
├── weights/               # Pre-trained weights (place here)
├── data/                  # Datasets (auto-downloaded where possible)
└── results/               # Evaluation outputs
```

## Weights

### ViViT-B (Kinetics-400 and EPIC-Kitchens)

Download the "ViViT Fact. Enc." weights from the [TAdaConv model zoo](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/MODEL_ZOO.md), then convert:

```bash
python scripts/convert/vivit.py <downloaded.pth> weights/vivit_b_kinetics400.pth configs/convert/vivit_b.txt
```

Fine-tuned temporal sub-model weights (used by some evaluation configs) are available [here](https://drive.proton.me/urls/12TW6GHZXW#hehlgPwql3ln).

### ViTDet-B (COCO → VID)

Download "Cascade Mask R-CNN, ViTDet, ViT-B" from [Detectron2 ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) and the VID fine-tuned weights from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) (`frcnn_vitdet_final.pth`), then convert:

```bash
python scripts/convert/vitdet.py <downloaded.pkl> weights/vitdet_b_coco.pth configs/convert/vitdet_b.txt
```

## Data

### Kinetics-400

The `Kinetics400` dataset class automatically downloads and prepares the validation set on first use. Data is stored under `data/kinetics400/`.

### EPIC-Kitchens

Manual download required:
- Videos → `data/epic_kitchens/videos/` from [here](https://drive.google.com/drive/folders/1OKJpgSKR1QnWa2tMMafknLF-CpEaxDbY)
- Labels `EPIC_100_train.csv` and `EPIC_100_validation.csv` → `data/epic_kitchens/` from [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

### ImageNet VID

Manual download required. Place `vid_data.tar` from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) at `data/vid/data.tar`. The dataset class handles extraction on first use.

## Evaluation

### TBKV ViViT on Kinetics-400

The main evaluation script is `scripts/evaluate/tbkv_vivit_kinetics400.py`. It uses `utils/test_evaluate.py` which implements the full two-pass TBKV evaluation with detailed per-block statistics.

#### Run TBKV evaluation

```bash
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=1000
```

For a quick smoke test on 25 videos:

```bash
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=25
```

> **Important:** `PYTHONUNBUFFERED=1` and `--no-capture-output` are required. Without them, `conda run` buffers all stdout until the process exits — you will see no output until the run completes.

#### Run vanilla baseline (for comparison)

```bash
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/vivit_kinetics400.py base n_items=1000 batch_size=4
```

The vanilla script supports `batch_size > 1` (videos are zero-padded within each mini-batch to the largest spatial size). TBKV must use `batch_size=1` because each video builds its own per-block KV cache that cannot be shared across batch items.

### Configuration

Configs are in `configs/evaluate/vivit_kinetics400/`. The config name passed on the command line (`tbkv`, `base`, etc.) selects a `.yml` file in that directory:

| Config | Description |
|--------|-------------|
| `base` | Vanilla ViViT, no caching |
| `tbkv` | TBKV merged cache (`local_merge_ratio=0.5`, `r_match=0.95`) |
| `tbkv_raw` | TBKV raw cache (no merging, higher recall) |
| `tbkv_tome` | TBKV with ToMe as the merge strategy |
| `temporal_24` | Temporal-only token pruning, 24 frames |

Any config key can be overridden directly on the command line:

```bash
# Sweep merge ratio
python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=500 \
  model.spatial_config.block_config.local_merge_ratio=0.3

# Sweep match threshold
python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=500 \
  model.spatial_config.block_config.r_match=0.90
```

### Live Progress

The evaluation logs a summary line every 10 videos so you can monitor progress in real time:

```
[ 10/1000]  Top-1: 80.0%  Top-5: 100.0%  Matching linear_flops: 2.241e+12
[ 20/1000]  Top-1: 80.0%  Top-5:  95.0%  Matching linear_flops: 2.284e+12
```

### Output

Results are written to `results/evaluate/vivit_kinetics400/<config_name>/`:

| File | Contents |
|------|----------|
| `output.txt` | Full report: caching stats, matching stats, per-block breakdown |
| `metrics.csv` | Top-1 and Top-5 accuracy |
| `counts.csv` | FLOPs breakdown (linear, matmul, add, bias) |

The `output.txt` report is structured as follows:

```
================
  CACHING PASS
================
  Frames used for caching : 16
  FLOPs breakdown (avg per video): ...

  Per-block stats (averaged over caching frames × videos):
    [spatial_model.backbone.blocks.0]
      avg foreground tokens         : 115.3
      avg background tokens         : 92.5
      cached tokens (total, final)  : 624
      avg merged tokens added/frame : 46.4
      avg merge factor (bg→merged)  : 1.99x
      cache K+V memory              : 46.006 MB

=================
  MATCHING PASS
=================
  Top-1 Accuracy              : 80.00%
  Top-5 Accuracy              : 100.00%
  FLOPs breakdown (avg per video): ...
  Matching algorithm FLOPs    : 8.76e+09
  KV FLOPs saved vs. baseline : 1.55e+10
  Net FLOPs savings           : 6.72e+09

  Per-block stats (averaged over matching frames × videos):
    [spatial_model.backbone.blocks.1]
      avg tokens matched (bg reused)  : 86.5
      avg cache entries used          : 47.7
      avg final tokens (fg+bg+cached) : 187.6
```

### Ablation Studies

For TBKV ViViT ablations on Kinetics-400, use `scripts/evaluate/vivit_kinetics400_ablation.py` with `tbkv_ablation`:

```bash
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/vivit_kinetics400_ablation.py tbkv_ablation \
  n_items=50 \
  local_merge_ratio=[0.5,0.75] \
  r_match=[0.75,0.95,1.0]
```

Quick smoke test:

```bash
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/vivit_kinetics400_ablation.py tbkv_ablation n_items=25
```

Then visualize saved sweep outputs with:

```bash
python scripts/evaluate/plot_from_sweep.py
python scripts/evaluate/plot_tbkv_flops_accuracy.py
```

To sweep the number of caching frames $P$:

```bash
python scripts/evaluate/sweep_p_frames.py
python scripts/evaluate/plot_p_frames.py
```

### Other Evaluation Tasks

**EPIC-Kitchens:**
```bash
python scripts/evaluate/vivit_epic_kitchens.py <config_name>
```

**ImageNet VID (ViTDet):**
```bash
# TBKV ViTDet
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/tbkv_vitdet_vid.py _tbkv n_items=5

# Vanilla ViTDet baseline (use base_672)
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/vitdet_vid.py base_672 n_items=5
```

`base_672` is the vanilla baseline config. `_base` is a shared base config fragment and is not intended as a standalone vanilla run target.

## Fine-Tuning

Some evaluation configs require a fine-tuned temporal sub-model. To fine-tune from scratch:

1. Run spatial pre-caching:
```bash
python scripts/spatial/vivit_epic_kitchens.py <config>
```

2. Run training:
```bash
python scripts/train/vivit_epic_kitchens.py <config>
```

This produces `weights/vivit_b_epic_kitchens_final_<N>.pth`.

## Troubleshooting

**No terminal output during evaluation**
```bash
# Always run with these flags:
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output python ...
```

**CUDA out of memory**
```bash
# TBKV is batch_size=1 by design. If OOM occurs, reduce n_cache_frames:
python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=100 frame_split=8
```

**Dataset download fails**
```bash
# Manually place Kinetics-400 validation frames under:
# data/kinetics400/val/<class_name>/<video_id>/frame_%06d.jpg
```

**ModuleNotFoundError**
```bash
# Ensure the repo root is on PYTHONPATH
export PYTHONPATH="$PYTHONPATH:."
conda activate eventful-transformer
python -c "from src.tbkv.tbkv_blocks import TBKVBlock; print('OK')"
```

## Code Style

Format with [Black](https://black.readthedocs.io/en/stable/) using the default 88-character line limit:

```bash
black <FILE>
```

## References

- **Eventful Transformers** (ICCV 2023): [paper](https://arxiv.org/abs/2308.13494) · [project page](https://wisionlab.com/project/eventful-transformers/)
- **ViViT**: Arnab et al., "ViViT: A Video Vision Transformer", ICCV 2021
- **ToMe**: Bolya et al., "Token Merging: Your ViT but Faster", ICLR 2023
- **TAdaConv** (ViViT weights): [model zoo](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/MODEL_ZOO.md)
- **Kinetics-400**: [DeepMind](https://www.deepmind.com/open-source/kinetics)

