# Persistent Semantic Memory for Efficient Video Transformer Inference
## Reproducibility archive

This archive contains the complete implementation of Persistent Semantic
Memory (PSM), the host integration it is layered on, the configuration files
pinning every hyperparameter used in the paper, and the scripts that compute
every reported metric. The implementation is complete: no part of the method is
hosted elsewhere. The datasets and the pretrained checkpoints are not
redistributed here; both are public and section 2 gives their sources.

The method is called `psm` throughout: `src/psm/` is the implementation,
`psm_eventful_filter_*` and `eventful_psm_*` are the configs, and
`eventful_psm` in a result directory name marks a run of it. The naming in the
code, the configs and the paper is the same.

---

## Quick start — checking the archive in minutes, without a GPU

All commands in this README are run from the archive root.

```bash
conda env create -f environment.yml
conda activate eventful-transformer

python tests/test_psm.py                          # 13 self-checks of the method code
python scripts/reproduce/verify_paper_results.py  # re-derives the reported detection values from results/
python scripts/plots/make_vitdet672_figures.py    # regenerates both paper figures
```

None of these three steps needs a GPU, a dataset or a weight file: the
method's behavior, the stored results and the figures are all checkable
offline. Re-running the evaluations themselves (section 5) needs all three.

---

## 1. Overview

PSM is a training-free, inference-time layer that adds content-addressable
reuse to an existing video-transformer accelerator. Given any host method that
proposes a set of tokens to recompute on each frame, PSM removes the fraction
of that set which is already explained by a compact prototype cache built
online from earlier frames, and forces the reduced set onto the host's gates.
No weights are modified or fine-tuned; with the reuse fraction set to zero the
pipeline is bit-for-bit the unmodified host.

**Where the method lives.** The paper's Algorithm 1 and the three method steps
map onto the source as follows.

| Paper | File | What is there |
|---|---|---|
| Algorithm 1, layered filter | `src/psm/blocks.py` | `_EventfulPSMMixin._compute_forced_index` — takes the host's top-`k` proposal, drops the cache-matched fraction, forces the kept set onto all three gates |
| Memory construction | `src/psm/blocks.py` (`_finalize_cache`) + `src/psm/prototypes.py` (`compute_merge`) | accumulates tokens over the caching window, then merges them into prototypes with `T` bipartite soft-matching passes |
| Memory retrieval | `src/psm/filter.py` (`psm_keep_mask`) | cosine similarity of every candidate against every prototype; ranks candidates by best match |
| Host (Eventful Transformers) | `src/core/blocks.py` | `EventfulBlock`, `EventfulTokenwiseBlock` — the unmodified host blocks PSM subclasses |

Only the layered path above is shipped. Every table and figure in the paper
comes from it, and nothing in `src/` is unreachable from the entry points —
`tests/test_psm.py` asserts this.

---

## 2. Datasets

Both benchmarks are public and standard. No dataset files are redistributed
here; `data/vid/` and `data/kinetics400/` are shipped empty.

**ImageNet VID** (ILSVRC2015 VID; Russakovsky et al. 2015). We use the
official ILSVRC2015 VID release. Evaluation uses the 639-video validation
split. Place the packaged archive at `data/vid/data.tar`; the loader unpacks
and prepares it on first use (`src/datasets/vid.py`).

**Kinetics-400** (Kay et al. 2017). We use the official CVDF/DeepMind Kinetics
distribution. Evaluation uses the 19,877-clip validation split; expect roughly
123 GB. Place it under `data/kinetics400/`; the loader unpacks and decodes it on
first use (`src/datasets/kinetics400.py`). That loader can also fetch the split
from the official distribution itself when constructed with `download=True`,
which is off by default — as with VID, the data is otherwise expected to be
present already and no network access takes place.

### Preprocessing

**Detection (ImageNet VID).** Frames are resized to a square input at two
resolutions: 672 x 672, giving 1,764 tokens per frame, and 1024 x 1024, giving
4,096 tokens per frame. The resize rule is shared by every detection script
(`VIDResize`, short side `640 * size // 1024`, so 672 -> 420 and 1024 -> 640),
so all methods see identical pixels at a given resolution.

**Recognition (Kinetics-400).** Factorized ViViT-B. Each clip is read as
32-frame views divided into sixteen 2-frame tubelet steps, with 12 views per
clip at 224 x 224, giving 197 tokens per step (196 spatial + 1 class token).

### Weights

Model weights are frozen pretrained checkpoints for every method, PSM included;
**nothing in this archive trains**. **No checkpoint is redistributed here**, and
`weights/` ships empty. Every checkpoint is a public release of either the host
method — Eventful Transformers (Dutson et al., ICCV 2023), cited in the paper —
or of the original backbone, and each is obtained from that public release. The
five files below are all that `weights/` needs.

Two of them are published under different parameter names and must be remapped
before use. The remap scripts and their name-mapping configs are part of the
host method's official code release, at the paths given in the commands below;
obtain that release and run the commands from its root.

**Detection — `weights/vitdet_b_vid.pth`.** Obtain `frcnn_vitdet_final.pth`,
the ViTDet-B detector fine-tuned on ImageNet VID, from the host method's
official release (Eventful Transformers, ICCV 2023), then remap it:

```bash
./scripts/convert/vitdet.py <obtained> weights/vitdet_b_vid.pth ./configs/convert/vitdet_b.txt
```

**Recognition, dense — `weights/vivit_b_kinetics400.pth`.** Obtain the
"ViViT Fact. Enc." Kinetics-400 weights from the TAdaConv model zoo
(Huang et al., TAdaConv), then remap them:

```bash
./scripts/convert/vivit.py <obtained> weights/vivit_b_kinetics400.pth ./configs/convert/vivit_b.txt
```

**Recognition, per-budget — `weights/vivit_b_kinetics400_final_{24,48,96}.pth`.**
The host method's fine-tuned temporal sub-models, one per token budget `k`.
Obtain them from the host method's official release; they need no remap.

**These three can equivalently be regenerated from scratch**, which is the
strongest reproducibility path and requires no distributed artifact at all: the
host method's public code release contains the exact training configurations
used to produce them, `configs/train/vivit_kinetics400/final_{24,48,96}.yml`.
Running those configs reproduces the three checkpoints directly.

> The dense checkpoint and the `k = 24` one are the **same model**: we verified
> that `vivit_b_kinetics400.pth` and `vivit_b_kinetics400_final_24.pth` hold
> bit-identical tensors (all 204, max difference 0), differing only in
> serialization. Fine-tuning at `k = 24` left the base weights unchanged; the
> `k = 48` and `k = 96` checkpoints do differ, in the temporal sub-model and
> classifier only (54 of 204 tensors). Either filename works for the dense row.

`bash scripts/run/00_check_setup.sh` verifies that all five files are present
and correctly named before any evaluation starts.

---

## 3. Environment

Results were produced on **one NVIDIA Quadro RTX 6000 (24 GB), CUDA 11.8**,
running the evaluation scripts sequentially on an idle device.

| | |
|---|---|
| Python | 3.10.19 |
| PyTorch | 2.0.1 (CUDA 11.8) |
| torchvision | 0.15.2 |
| torchmetrics | 0.11.4 |
| detectron2 | 0.6 |
| numpy | 1.23.5 |
| omegaconf | 2.3.0 |
| einops | 0.8.2 |
| matplotlib | 3.7.2 |
| Pillow | 9.5.0 |
| scipy | 1.10.1 |
| tqdm | 4.65.0 |
| PyYAML | 6.0.3 |

```bash
conda env create -f environment.yml
conda activate eventful-transformer
```

`requirements.txt` carries the same pins for a pip-only install. Two notes:

- **`torchmetrics==0.11.4` is load-bearing.** VID mAP@50 comes from its
  `MeanAveragePrecision`, whose internals changed in 1.x. Floating this pin
  changes the reported detection numbers.
- **detectron2 0.6 is built from source** and needs `gcc >= 5.4` and
  `cuda-nvcc`. It is the most common install failure; if `import detectron2`
  fails, every detection result fails with it.

**Accuracy and GFLOPs are deterministic and hardware-independent** — GFLOPs are
counted by in-code operation counters, not estimated, and the count includes
PSM's own matching cost. **Latency and peak memory are hardware-dependent** and
are only valid on an idle GPU; every run script warns if the device is busy.

---

## 4. Hyperparameters

The final settings used in the paper, in the notation of the paper and of the
code.

| Paper | Code | Detection (VID) | Recognition (K400) |
|---|---|---|---|
| Reuse fraction γ | `cache_reuse` | **0.25** | **0.95** |
| Merge passes T | `merge_iterations` | **6** | **2** |
| Host token budget k | `token_top_k` | **512** | **24, 48, 96** |
| Warm-up / caching window W | `warmup` (VID), `frame_split` (K400) | **4** frames | **4** temporal steps |
| Merge ratio per pass | `merge_ratio` | 0.5 | 0.5 |
| Protocol | `replay_matching` | true (two-pass replay) | false (single pass) |

**On the similarity threshold τ.** The paper's Memory Retrieval subsection
presents retrieval as a threshold test against τ. The implementation used for
every reported number realises the same decision by **rank rather than by a
fixed threshold**: `psm_keep_mask` (`src/psm/filter.py`) scores each of the
host's `k` candidates by its best cosine similarity to any prototype and keeps
the `k - round(γk)` least similar, so the number of reused tokens is exactly
`⌊γk⌋` as stated in Algorithm 1. **There is therefore no numeric τ to report
for the paper's runs** — γ is the only knob controlling how much is reused. The
class token is never dropped (`protect` in the same function).

**On β and the saliency split.** The layered method reported in the paper
does not split tokens by saliency: every candidate the host proposes is matched
against the memory on equal terms, with only the class token protected.
**No β therefore takes a value in any reported result**, and the code
implementing such a split is not part of this archive.

---

## 5. Reproducing each table and figure

Driver scripts under `scripts/run/` wrap the commands below with skip guards, a
GPU-busy warning and the expected output paths. Run either the driver or the
raw command; they are equivalent. The drivers change to the archive root
themselves; raw commands must be issued from it.

```bash
bash scripts/run/00_check_setup.sh    # pre-flight: environment, weights, data, GPU
```

Each script is idempotent — a step whose output already exists is skipped, so
an interrupted run can be restarted. `FORCE=1` redoes finished steps.

### Figures 1 and 2

```bash
python scripts/plots/make_vitdet672_figures.py     # or: bash scripts/run/06_figures.sh
```

Writes `fig_vid672_frontier.pdf` (Figure 1) and `fig_vid672_ablation.pdf`
(Figure 2), plus a `.png` of each, to `paper/Figures/`. That directory is not
shipped: the script creates it on first run, relative to its own location, so
this works from a fresh unzip and from any working directory. This needs **no
GPU, no weights and no dataset** — it takes seconds and is the fastest way to
confirm the archive runs.

> **Disclosure — the plotting script holds its values as constants.**
> `make_vitdet672_figures.py` carries the plotted numbers inline (`METHODS` and
> `ABL` near the top of the file) instead of reading `results/` at plot time, so
> re-running an evaluation does not change a figure until those constants are
> edited. The constants are **not** independent of the runs: every one of them
> is cross-checked against the stored `results/` outputs by
> `python scripts/reproduce/verify_paper_results.py`, which re-derives each
> value from the corresponding `output.txt` and fails on any mismatch. Running
> that script is therefore the way to confirm the figures against the data, and
> the way to detect that a constant needs updating after a re-run.

### Table 1 — ImageNet VID, ViTDet-B, both resolutions

```bash
bash scripts/run/01_vid_672.sh      # 672 block
bash scripts/run/02_vid_1024.sh     # 1024 block
```

Full 639-video split, every gated method at the common budget `k = 512`. The
individual commands, at 672 (swap `_672` for `_1024` for the other block):

```bash
python scripts/evaluate/vitdet_vid.py base_672                                  # dense
python scripts/evaluate/vitdet_vid.py temporal_672 token_top_k=[512]            # Eventful
python scripts/evaluate/vitdet_vid.py spatiotemporal_672 token_top_k=[512]      # + spatial pool
python scripts/evaluate/vitdet_vid.py stgt_672 token_top_k=[512]                # STGT
python scripts/evaluate/maskvd_vitdet_vid.py _output=results/evaluate/vitdet_vid/maskvd_672/
python scripts/evaluate/psm_eventful_vitdet_vid.py psm_eventful_filter_672 \
    token_top_k=[512] cache_reuse=0.25 merge_iterations=6 warmup=4 \
    replay_matching=true measure_latency=true                                   # PSM (layered)
```

Two traps:

1. **`measure_latency=true` is mandatory for the PSM rows.** Unlike
   `vitdet_vid.py`, `psm_eventful_vitdet_vid.py` gates its latency and
   peak-memory harness behind that flag and silently omits both columns
   without it.
2. **Latency and memory print to stdout and are not saved.** Tee the script if
   you want to keep them.

`k` is held fixed in *absolute* terms across resolutions, so it is 29% of the
1,764 tokens at 672 but only 12.5% of the 4,096 tokens at 1024. That is
deliberate, and is why every gated method loses accuracy at 1024 while the
dense backbone gains.

The MaskVD row requires the third-party MaskVD implementation (cited in the
paper) on the path; the other five rows need nothing outside this archive.

### Table 2 — Kinetics-400 reuse-fraction ablation

```bash
bash scripts/run/05_k400_ablation.sh
```

```bash
for cr in 0.25 0.5 0.75 0.95; do
  python scripts/evaluate/eventful_psm_vivit_kinetics400.py eventful_psm \
      cache_reuse=$cr n_items=1988
done
```

Fixed 10% subsample (1,988 clips) at host budget `k = 24`.

### Table 3 — Kinetics-400, full split

```bash
bash scripts/run/04_k400_main.sh
```

```bash
python scripts/evaluate/vivit_kinetics400.py base                        # dense ViViT-B
python scripts/evaluate/eventful_psm_vivit_kinetics400.py eventful_psm_24   # Eventful, k=24
python scripts/evaluate/eventful_psm_vivit_kinetics400.py eventful_psm_24 \
    cache_reuse=0.95 merge_iterations=2                                  # + PSM, k=24
```

and likewise for `eventful_psm_48` and `eventful_psm_96`. Full 19,877-clip
split, single-pass accounting: each clip is processed once, the first four
temporal steps in caching mode and the remaining twelve in matching mode, with
every operation of both phases counted. Outputs report
`caching A + matching B = total C` GFLOPs per clip; the paper's "Whole clip"
column is C and "Matching" is B.

> **Naming trap.** The configs `eventful_psm_24/48/96` default to
> `cache_reuse=0.0`, which disables the PSM filter entirely — run bare, they
> *are* the Eventful baseline rows. The PSM rows are the same configs with
> `cache_reuse=0.95 merge_iterations=2` passed on the command line.

### Figure 2 grid — detection ablation

```bash
bash scripts/run/03_vid_ablation.sh
```

Full 4x4 γ x T grid on the fixed 10% detection subsample (64 of 639 videos),
plus the dense reference on the same 64 videos:

```bash
python scripts/evaluate/vitdet_vid.py base_672 n_items=64
for cr in 0.25 0.5 0.75 0.95; do for mi in 2 4 6 8; do
  python scripts/evaluate/psm_eventful_vitdet_vid.py psm_eventful_filter_672 \
      token_top_k=[512] warmup=4 cache_reuse=$cr merge_iterations=$mi \
      n_items=64 replay_matching=true
done; done
```

> **Override order matters.** Overrides appear in the output directory name in
> the order they are passed. Reordering the arguments produces a
> differently-named directory that the plotting and verification scripts will
> not find.

### Checking the archive itself

```bash
python tests/test_psm.py
```

Thirteen self-checks that need no GPU, no dataset and no weights — among them:
that the reuse fraction drops exactly ⌊γk⌋ candidates, that γ = 0 is a no-op,
that an exact cache match is the token dropped, that the class token survives
any γ, that `T` merge passes shrink the prototype set monotonically, that PSM
subclasses the unmodified host, that all 15 reported configs resolve, and that
every file under `src/` is reachable from the entry points. Run this first — it
takes seconds and catches a broken environment before a multi-hour evaluation
does.

### Checking a finished run

```bash
python scripts/reproduce/verify_paper_results.py
```

Re-derives each reported 672 detection value from the stored `output.txt`,
cross-checks the saved `config.yml`, and confirms the split size. Runs no model.

---

## 6. Config-to-result map

Detection configs live in `configs/evaluate/vitdet_vid/`, recognition configs in
`configs/evaluate/vivit_kinetics400/`. Overrides are given where the config
alone does not pin the reported cell.

| Paper cell | Config | Overrides |
|---|---|---|
| Table 1, dense | `base_672` / `base_1024` | — |
| Table 1, Eventful | `temporal_672` / `temporal_1024` | `token_top_k=[512]` |
| Table 1, + spatial pool | `spatiotemporal_672` / `spatiotemporal_1024` | `token_top_k=[512]` |
| Table 1, STGT | `stgt_672` / `stgt_1024` | `token_top_k=[512]` |
| Table 1, MaskVD | `scripts/evaluate/maskvd_vitdet_vid.py` | `input_size=672` (default) |
| Table 1, PSM (layered) | `psm_eventful_filter_672` / `psm_eventful_filter_1024` | `token_top_k=[512] cache_reuse=0.25 merge_iterations=6 warmup=4 replay_matching=true measure_latency=true` |
| Table 2, γ sweep | `eventful_psm` | `cache_reuse=<γ> n_items=1988` |
| Table 3, dense | `base` | — |
| Table 3, Eventful | `eventful_psm_{24,48,96}` | — (config default `cache_reuse=0.0`) |
| Table 3, + PSM | `eventful_psm_{24,48,96}` | `cache_reuse=0.95 merge_iterations=2` |
| Figure 1 | same runs as Table 1, 672 block | — |
| Figure 2 | `psm_eventful_filter_672` | `token_top_k=[512] warmup=4 cache_reuse=<γ> merge_iterations=<T> n_items=64 replay_matching=true` |

**Latency and peak memory** are measured on every Table 1 row. For the dense,
Eventful, + spatial pool, STGT and MaskVD rows the scripts
(`vitdet_vid.py`, `maskvd_vitdet_vid.py`) measure and print them on every run
with no flag; the PSM rows are the only ones whose script gates the harness
behind `measure_latency=true`, which is why that override appears in the PSM
rows above and must not be dropped. Table 2, Table 3 and the figures report
no latency.

> **Reading a saved `config.yml`.** Each result directory stores the merged
> configuration, written when the run starts. Command-line overrides appear at
> **top level** — those are the authoritative values for the run. The nested
> `model.backbone_config.block_config` section still shows the config *file's*
> defaults (e.g. `cache_reuse: 0.5`, `merge_iterations: 4`), because the
> overrides are routed into the blocks after the file is written. For the PSM
> rows, read the top-level `cache_reuse` / `merge_iterations`; the directory
> name records the same values, and `verify_paper_results.py` checks them.

Weight files, as referenced by the configs. Place them under `weights/`;
section 2 gives the download and conversion steps for each:

| File | Used by |
|---|---|
| `weights/vitdet_b_vid.pth` | every detection result |
| `weights/vivit_b_kinetics400.pth` | dense ViViT-B reference |
| `weights/vivit_b_kinetics400_final_24.pth` | K400 at k=24 |
| `weights/vivit_b_kinetics400_final_48.pth` | K400 at k=48 |
| `weights/vivit_b_kinetics400_final_96.pth` | K400 at k=96 |

---

## 7. Metrics

**Detection.** mAP@50 via `torchmetrics.detection.MeanAveragePrecision`, over
the full 639-video validation split.

**Recognition.** Top-1 and Top-5 over the full 19,877-clip validation split
(`TopKAccuracy` in `src/utils/misc.py`).

**Cost.** GFLOPs come from in-code operation counters (`src/core/counting.py`),
so every reported reduction already includes PSM's own matching cost. Latency
is CUDA-event wall-clock time and memory is peak GPU allocation, both measured
under an identical warm-up protocol.

---

## 8. Contents

```
.
├── README.md
├── environment.yml            conda environment, pinned
├── requirements.txt           same pins, pip
├── data/
│   ├── vid/                   place data.tar here (shipped empty)
│   └── kinetics400/           place Kinetics-400 here (shipped empty)
├── weights/                   place the five checkpoints here (shipped empty)
├── src/
│   ├── psm/                   the method
│   │   ├── blocks.py          Algorithm 1 (recompute-set filter)
│   │   ├── filter.py          memory retrieval, keep/reuse decision
│   │   ├── prototypes.py      memory construction (compute_merge)
│   │   └── merge.py           bipartite soft-matching primitive
│   ├── core/                  host blocks, backbones, policies, FLOP counters
│   ├── models/                ViTDet, ViViT, and the PSM-enabled ViViT
│   ├── datasets/              VID and Kinetics-400 loaders and preprocessing
│   └── utils/                 config resolution, evaluation loop, metrics
├── utils/                     shared evaluation helpers
├── tests/test_psm.py          self-checks (no GPU, no data, no weights)
├── configs/
│   ├── evaluate/vitdet_vid/       detection configs
│   ├── evaluate/vivit_kinetics400/ recognition configs
│   ├── models/                     model definitions
│   └── detectron/                  detector definitions
├── scripts/
│   ├── evaluate/              one script per reported method
│   ├── plots/                 Figure 1 and Figure 2
│   ├── reproduce/             offline verification of stored results
│   └── run/                   driver scripts, one per table/figure
└── results/evaluate/          raw outputs of the runs behind the paper
```

`results/evaluate/` holds each run's saved `config.yml` and raw `output.txt`
(plus `metrics.csv`/`counts.csv`, or `results.txt` for the MaskVD and
EventfulPSM harnesses, where the script writes them), so the reported values
can be inspected and re-derived without a GPU. The bare `temporal_672/` directory is the six-budget
Eventful sweep; the k = 512 block of its `output.txt` backs the Table 1 row and
the k = 384 block backs the corresponding claim in the paper text (both are
cross-checked by `verify_paper_results.py`).
