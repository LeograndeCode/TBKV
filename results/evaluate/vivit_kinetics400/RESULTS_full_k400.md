# ViViT-B / Kinetics-400 — full-validation results (canonical reference)

Dataset: Kinetics-400 **validation**, **19,877 clips** (full split).
All numbers verified from `results/evaluate/vivit_kinetics400/<dir>/output.txt`
on 2026-07-25. This file is the single source of truth for the K400 numbers —
the config names are confusing, so the mapping below is authoritative.

## Naming, once and for all
- **`k` = 24 / 48 / 96** is Eventful's **token budget**: how many of the ~197
  spatial tokens Eventful recomputes per frame (higher k = more accurate, more
  expensive). Each k also loads its own fine-tuned checkpoint
  (`weights/vivit_b_kinetics400_final_{24,48,96}.pth`).
- **`cr` (a.k.a. cache_reuse / γ)** = the fraction of Eventful's proposed
  recompute set that the TempoMem filter **drops** (matches to the cache and
  skips). `cr = 0.0` ⇒ filter disabled ⇒ **this is plain Eventful** (verified
  bit-identical to genuine `temporal_24` on 25 clips). `cr > 0` ⇒ **TempoMem**.
- Config-name trap: the **bare** `eventful_tbkv` is k=24 **cr=0.5** (TempoMem);
  the numbered `eventful_tbkv_24/48/96` default to **cr=0.0** (Eventful).

## Dense baseline
| Top-1 | Top-5 | GFLOPs/clip |
|---|---|---|
| 78.64% | 93.60% | ~3360 |
`base/` (whole-clip dense forward).

## Eventful (cr = 0.0) — the host, by budget
| k | Top-1 | Top-5 | GFLOPs/clip (caching + matching) | dir |
|---|---|---|---|---|
| 24 | 62.38% | 82.23% | 618 + 435 = 1053 | `eventful_tbkv_24` |
| 48 | 67.54% | 87.26% | 1016 + 860 = 1876 | `eventful_tbkv_48` |
| 96 | **75.71%** | **92.35%** | 1814 + 1711 = 3525 | `eventful_tbkv_96` |

(These are the `cr=0.0` TBKV-proxy, bit-identical to genuine Eventful. No
genuine `temporal_24/48/96` full run exists — only deleted subset runs.)

## Layered TempoMem (cr > 0)
| k | cr (γ) | protocol | Top-1 | Top-5 | GFLOPs/clip | dir |
|---|---|---|---|---|---|---|
| 24 | 0.5 | non-replay | 59.85% | 79.83% | 618 + 222 = 840 | `eventful_tbkv` |
| 24 | 0.95 | non-replay | 59.82% | 79.83% | 618 + 27 = 645 | `eventful_tbkv_24-cache_reuse=0.95-merge_iterations=2` |
| 48 | 0.5 | non-replay | 58.37% | 78.62% | 1016 + 435 = 1451 | `eventful_tbkv_48_cr05` |
| 48 | 0.95 | non-replay | 58.37% | 78.61% | 1016 + 45 = 1062 | run with `max_tars=20`; not in local results (numbers supplied 2026-07-25) |
| 48 | 0.95 | **replay** | 58.45% | 78.58% | matching 45 (caching excluded) | `eventful_tbkv_48-cache_reuse=0.95-merge_iterations=2-replay_matching=true` |
| 96 | any | — | — | — | — | **not run** |

Reproduce the non-replay k=48/cr=0.95 point with:
`python scripts/evaluate/eventful_tbkv_vivit_kinetics400.py eventful_tbkv_48 cache_reuse=0.95 merge_iterations=2`
(`max_tars=20` just selects the val_20 shards = full 19,877 clips.)

The two k=48/cr=0.95 rows differ ONLY in protocol: non-replay counts the
1016 caching term (total 1062, comparable to the other non-replay rows and to
Eventful k=48's 1876 → a 43% whole-clip cut); replay excludes it (matching 45).
Note the non-replay cr=0.95 Top-1 (58.37%) is *identical* to cr=0.5 (58.37%) —
a clean confirmation of the flat step (matching 435 → 45 for the same accuracy).

Notes:
- **Protocol matters**: most points are *non-replay* (caching + matching both
  counted). The one k=48/cr=0.95 point is *replay* (caching discarded/excluded)
  — its "matching 45" is NOT directly comparable to the "caching + matching"
  totals of the others.
- **Accuracy is a step, not a slope**: at fixed k, accuracy drops once at the
  first reuse and is then flat across cr (k=24: 59.85 → 59.82 from cr=0.5→0.95;
  k=48: 58.37 → 58.45). The matching term keeps falling because it tracks the
  drop *count*, but the whole-clip cost is dominated by the shared caching term.
- **TempoMem's advantage is budget-specific**: at k=48, TempoMem@cr=0.5
  (58.37% at matching 435) is *dominated* by Eventful@k=24 (62.38% at the same
  matching 435). Only k=24 is a favourable operating point.

## Standalone TempoMem (no host) — DO NOT trust the GFLOPs
| variant | Top-1 | Top-5 | matching GFLOPs | net FLOP saving | dir |
|---|---|---|---|---|---|
| raw cache | 76.00% | 92.15% | 3165.6 | −30.7 (negative) | `tbkv_raw-weights=weights/vivit_b_kinetics400.pth` |
| merged (replay) | 76.47% | 92.38% | 3020.2 | +2.7 | `tbkv_merged_replay_full` |

**KNOWN BUG (documented in `utils/evaluate.py`):** the standalone ViViT caching
pass is inflated to a full multi-view forward (~dense, ~3360 GFLOPs) regardless
of frame count, and the per-view cache forces caching and matching to use the
same 12 views. So the "matching-only" metric is NOT a real saving for the
standalone ViViT path — per clip you pay caching + matching ≈ 2× dense. The
accuracy numbers are fine; the compute numbers are not usable as savings.
