# Method comparison — base vs Eventful vs Eventful-TBKV vs MaskVD

## ViViT / Kinetics-400 — 100 videos

| method | Top-1 | Top-5 | GFLOPs | notes |
|---|---|---|---|---|
| base (dense) | 73.0% | 96.0% | 3355 /clip | whole-clip, single pass |
| Eventful (temporal_24) | 71.0% | 95.0% | 613 /clip | whole-clip, token_top_k=24 |
| Eventful-TBKV `cache_reuse=0` | 61.0% | 82.0% | 435 /clip matching (27.2 /frame) | = pure Eventful under the caching/matching protocol |
| **Eventful-TBKV `cache_reuse=0.5`** | **59.0%** | **79.0%** | **222 /clip matching (13.9 /frame)** | real Eventful + TBKV recompute-set filter |

Reading:
- Eventful-TBKV is real Eventful with TBKV strictly on top (drops cache-matched
  tokens from Eventful's recompute set). With `cache_reuse=0.5` it **halves the
  matching-pass FLOPs** (435 -> 222 /clip, 27.2 -> 13.9 /frame) for **-2 Top-1**.
- The absolute Top-1 (59-61%) is below Eventful's canonical 71% because this
  protocol scores only the matching frames (warm-up frames excluded), not the
  whole clip. That protocol gap applies equally to every Eventful-TBKV row, so
  within-block comparisons are fair; cross-comparison to the 71% is not.
- Caching pass (warm-up, 618 GFLOPs/clip) is excluded from the matching FLOPs, on
  the premise it amortises in a streaming deployment.

## ViTDet / ImageNet-VID — 25 videos  (DONE)

Eventful and Eventful-TBKV sweep token_top_k = {128,256,384,512,768,1024}, so
each is a 6-point accuracy/FLOPs curve.

**Fair, apples-to-apples comparison (both matching-frame only, same warm-up
protocol): Eventful-TBKV `cr=0` (= exact Eventful) vs `cr=0.5` (ours):**

| token_top_k | Eventful cr=0  mAP@50 / GF/frame | Eventful-TBKV cr=0.5  mAP@50 / GF/frame |
|---|---|---|
| 128  | 0.786 / 1.7 | 0.655 / 1.1 |
| 256  | 0.865 / 2.9 | 0.716 / 1.7 |
| 384  | 0.894 / 4.1 | 0.759 / 2.3 |
| 512  | 0.903 / 5.3 | 0.769 / 2.9 |
| 768  | 0.905 / 7.8 | 0.789 / 4.1 |
| 1024 | 0.905 / 10.2 | 0.762 / 5.3 |

Reference points (WHOLE-STREAM per-frame accounting incl. keyframes — NOT
matching-only, so not directly comparable to the two columns above):
- Base (dense ViTDet): mAP@50 0.906, 174.5 GFLOPs/frame
- MaskVD: mAP@50 0.899, 85.3 GFLOPs/frame

VERDICT (ViTDet): on detection the recompute-set filter is **dominated by
Eventful** — at matched FLOPs Eventful has higher mAP@50 (e.g. at ~1.7 GF/frame:
Eventful 0.786 vs ours 0.716), and at matched mAP@50 Eventful is cheaper. The
~2x FLOP cut TBKV gives (e.g. 7.8->4.1 at k=768) costs ~11 pts mAP@50 (0.905->
0.789). This is the OPPOSITE of the ViViT result: on classification the class
token absorbs dropped patch tokens cheaply, but detection needs the spatial
tokens TBKV drops, so localization degrades. Honest negative result on ViTDet.

## Whole-Kinetics-400 val (19877 clips) — Eventful baselines (cr=0)

Full-val (not sampled) Eventful points at different token budgets, matching-frame
FLOPs. These are the clean Pareto anchors for the ViViT plot.

| config | token_top_k | Top-1 | Top-5 | GFLOPs/frame (matching) |
|---|---|---|---|---|
| eventful_tbkv_24 | 24 | 62.38 | 82.23 | 27.2 |
| eventful_tbkv_48 | 48 | 67.54 | 87.26 | 53.8 |

## Sweep: cache_reuse x merge_iterations

### ViViT / Kinetics-400 (100 clips). Matching-frame FLOPs.

| cache_reuse | Top-1 | Top-5 | GFLOPs/frame |
|---|---|---|---|
| 0.0 (=Eventful) | 61.0 | 82.0 | 27.2 |
| 0.25 | 59.0 | 79.0 | 20.6 |
| 0.50 | 59.0 | 79.0 | 13.9 |
| 0.75 | 59.0 | 79.0 | 7.2 |

merge_iterations in {2,4,8} gave IDENTICAL results at every cache_reuse (rows
collapsed above). Two clean findings:
(1) Accuracy is FLAT at 59.0/79.0 for all cache_reuse >= 0.25 while FLOPs fall
    27.2 -> 7.2 (3.8x). TBKV drops up to 75% of eventful tokens for a fixed
    2-point Top-1 cost vs exact Eventful.
(2) merge_iterations is inert here: once the prototype pool exists, how finely
    it is compressed does not change which tokens get dropped. The prototypes
    are robust to merge granularity.

### ViTDet / ImageNet-VID (25 videos, token_top_k=512). Matching-frame FLOPs
### (counting bug fixed; earlier ViTDet GFLOPs were ~10x undercounted).

| cache_reuse | merge_iters | mAP@50 | GFLOPs/frame |
|---|---|---|---|
| 0.0 (=Eventful) | - | 0.903 | 60.3 |
| 0.25 | 2 | 0.869 | 46.6 |
| 0.25 | 4 | 0.864 | 46.6 |
| 0.25 | 8 | 0.852 | 46.6 |
| 0.50 | 2 | 0.778 | 32.9 |
| 0.50 | 4 | 0.769 | 32.9 |
| 0.50 | 8 | 0.747 | 32.9 |
| 0.75 | 2 | 0.593 | 19.3 |
| 0.75 | 4 | 0.581 | 19.3 |

Unlike ViViT, on detection: (1) mAP@50 falls monotonically with cache_reuse
(0.903 -> 0.593), and (2) merge_iterations matters and FEWER is better (finer
prototypes = better matches). TBKV trades accuracy for FLOPs here; it does not
sit on Eventful's Pareto front. Root cause: detection needs the spatial tokens
TBKV drops (localization), whereas ViViT's class-token readout does not.

## Appendix: motion-compensation variant — robustness thesis REFUTED

An earlier TBKV-on-Eventful variant (motion-compensated reference, no token
dropping) was tested for robustness under frame stride on ViTDet/VID, n=25,
full sweep in results/sweeps/motion_tbkv_eventful.csv:

| stride | dense mAP@50 | Eventful | +motion-comp TBKV |
|---|---|---|---|
| 1  | 0.906 | 0.785 | 0.786 |
| 4  | 0.902 | 0.665 | 0.669 |
| 8  | 0.901 | 0.629 | 0.623 |
| 16 | 0.900 | 0.538 | 0.540 |

Eventful and the motion-comp variant degrade identically at every stride
(differences <= 0.6 pt, noise). Content-addressed local matching of stale
tokens buys no robustness; that variant is abandoned. The recompute-set
filter (the rows above) is the surviving Eventful-TBKV design.
