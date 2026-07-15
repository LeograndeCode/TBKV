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

## ViTDet / ImageNet-VID — 25 videos  (RUNNING, ~3-4 h)

| method | mAP | mAP@50 | GFLOPs/frame |
|---|---|---|---|
| Eventful (temporal_672) | — | — | — |
| Eventful-TBKV filter `cache_reuse=0` (protocol-matched Eventful) | — | — | — |
| Eventful-TBKV filter `cache_reuse=0.5` | — | — | — |
| MaskVD | — | — | — |

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
