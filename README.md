## Overview

This is the PyTorch code for our ICCV 2023 paper "Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers." Please see our [paper webpage](https://wisionlab.com/project/eventful-transformers/) and the [arXiv paper](https://arxiv.org/abs/2308.13494).

## Disclaimer

This is research-grade code, so it's possible you will encounter some hiccups. [Contact me](https://github.com/mattdutson/) if you encounter problems or if the documentation is unclear, and I will do my best to help.

## TLDR

Most of the interesting code (implementation of our core contributions) is in `eventful_transformer/blocks.py`, `eventful_transformer/modules.py`, and `eventful_transformer/policies.py`.

## Dependencies

Dependencies are managed using Conda. The environment is defined in `environment.yml`.

To create the environment, run:
```
conda env create -f environment.yml
```
Then activate the environment with:
```
conda activate eventful-transformer
```

## Running Scripts

Scripts should be run from the repo's base directory.

Many scripts expect a `.yml` configuration file as a command-line argument. These configuration files are in `configs`. The structure of the `configs` folder is set to mirror the structure of the `scripts` folder. For example, to run the `base_672` evaluation for the ViTDet VID model:
```
./scripts/evaluate/vitdet_vid.py ./configs/evaluate/vitdet_vid/base_672.yml
```

## Weights

Weights for the ViViT action recognition model (on Kinetics-400 and EPIC-Kitchens) are available [here](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/MODEL_ZOO.md). We use the "ViViT Fact. Enc." weights.

Weights for the ViTDet object detection model (on COCO) are available [here](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet). We use the "Cascade Mask R-CNN, ViTDet, ViT-B" weights. Weights on ImageNet VID are available [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) (`frcnn_vitdet_final.pth`).

The weight names need to be remapped to work with this codebase. To remap the ViViT weights, run:
```
./scripts/convert/vivit.py <old_weights> <new_weights> ./configs/convert/vivit_b.txt
```
with `<old_weights>` and `<new_weigtht>` replaced by the path of the downloaded weights and the path where the converted weights should be saved, respectively.

To remap the ViTDet weights, run:
```
./scripts/convert/vitdet.py <old_weights> <new_weights> ./configs/convert/vitdet_b.txt
```

Some ViViT evaluation scripts assume a fine-tuned temporal sub-model. Fine-tuned weights can be downloaded [here](https://drive.proton.me/urls/12TW6GHZXW#hehlgPwql3ln).

Alternatively, you can run the fine-tuning yourself. To do this, run a `spatial` configuration (to cache the forward pass of the spatial sub-model), followed by a `train` configuration. For example:
```
./scripts/spatial/vivit_epic_kitchens.py ./configs/spatial/vivit_epic_kitchens/50.yml
```
then
```
./scripts/train/vivit_epic_kitchens.py ./configs/train/vivit_epic_kitchens/final_50.yml
```
This will produce `weights/vivit_b_epic_kitchens_final_50.pth`.

## Data

The `datasets` folder defines PyTorch `Dataset` classes for Kinetics-400, VID, and EPIC-Kitchens.

The Kinetics-400 class will automatically download and prepare the dataset on first use.

VID requires a manual download. Download `vid_data.tar` from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) and place it at `./data/vid/data.tar`. The VID class will take care of unpacking and preparing the data on first use.

EPIC-Kitchens also requires a manual download. Download the videos from [here](https://drive.google.com/drive/folders/1OKJpgSKR1QnWa2tMMafknLF-CpEaxDbY) and place them in `./data/epic_kitchens/videos`. Download the labels `EPIC_100_train.csv` and `EPIC_100_validation.csv` from [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations) and place them in `./data/epic_kitchens`. The EPICKitchens class will prepare the data on first use.

## Other Setup

Scripts assume that the current working directory is on the Python path. In the Bash shell, run
```
export PYTHONPATH="$PYTHONPATH:."
```
Or in the Fish shell:
```
set -ax PYTHONPATH .
```

## Code Style

Format all code using [Black](https://black.readthedocs.io/en/stable/). Use a line limit of 88 characters (the default). To format a file, use the command:
```
black <FILE>
```

## TBKV Evaluation

The `scripts/evaluate/tbkv_vivit_kinetics400.py` script evaluates the TBKV-enhanced ViViT model on Kinetics-400. It uses a two-pass strategy: a **caching pass** over the first frames of each video to build a compressed KV cache, followed by a **matching pass** over the remaining frames that reuses those cached KV pairs.

### Quick start

Make sure Kinetics-400 validation data is in `data/kinetics400/` (the dataset class downloads it automatically on first use) and that the converted weights are at `weights/vivit_b_kinetics400.pth`.

Run the evaluation with live progress output:
```bash
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=1000
```

To evaluate on a smaller subset (e.g. for a smoke test):
```bash
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=25
```

> **Note:** `PYTHONUNBUFFERED=1` and `--no-capture-output` are required to see live terminal output. Without them, `conda run` buffers all stdout until the process exits.

### Comparing against the vanilla baseline

To run the standard (non-TBKV) ViViT for direct comparison:
```bash
PYTHONUNBUFFERED=1 conda run -n eventful-transformer --no-capture-output \
  python scripts/evaluate/vivit_kinetics400.py base n_items=1000 batch_size=4
```

The vanilla script supports `batch_size > 1` (videos are zero-padded to the largest spatial size in each mini-batch). TBKV must use `batch_size=1` because each video builds its own per-block KV cache.

### Configuration

Both scripts read their model and dataset configuration from `configs/evaluate/vivit_kinetics400/`. The config name (`tbkv`, `base`, `tbkv_raw`, etc.) corresponds to a `.yml` file in that directory. Key configs:

| Config | Description |
|--------|-------------|
| `base` | Vanilla ViViT, no caching |
| `tbkv` | TBKV with merged cache (`local_merge_ratio=0.5`, `r_match=0.95`) |
| `tbkv_raw` | TBKV with raw cache (no token merging) |
| `tbkv_tome` | TBKV using ToMe as the merge strategy |

Any config key can be overridden on the command line:
```bash
python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=500 model.spatial_config.block_config.local_merge_ratio=0.3
```

### Output

Results are written to `results/evaluate/vivit_kinetics400/<config_name>/`:

- `output.txt` — full evaluation report printed to terminal (caching pass stats, matching pass stats, per-block breakdown)
- `metrics.csv` — Top-1 and Top-5 accuracy
- `counts.csv` — FLOPs breakdown (linear, matmul, add, bias)

The terminal report includes live per-video progress every 10 videos:
```
[ 10/1000]  Top-1: 80.0%  Top-5: 100.0%  Matching linear_flops: 2.241e+12
[ 20/1000]  Top-1: 80.0%  Top-5:  95.0%  Matching linear_flops: 2.284e+12
```

### Sample results (25 videos)

| | Vanilla ViViT | TBKV ViViT (matching pass) |
|---|---|---|
| Top-1 | ~66–72% | ~80–84% *(sampling noise at n=25)* |
| Top-5 | ~91–97% | ~96–100% |
| `linear_flops` | 3.218e+12 | ~2.28e+12 (**−29%**) |
| `matmul_flops` | 1.374e+11 | ~7.97e+10 (**−42%**) |
| Net KV savings/video | — | ~7.15e+09 |

> Accuracy differences at small sample sizes are due to sampling variance. Run with `n_items=1000` or more for stable accuracy estimates.
