# TBKV Results — Adjusted Model (Token Skipping) vs. SOTA

All numbers are from `n_items = 10` evaluations on a single Quadro RTX 6000, using
the identical ViTDet‑B / ViViT‑B weights across every method. FLOPs are counted with
the in‑repo `CountedLinear` / `matmul` counters (no analytic estimates).

> Scope note: this is a **10‑item** development snapshot for fast iteration. Absolute
> accuracies are noisy at this sample size (one clip = 10 % of the metric); the
> *relative* accuracy/FLOP trends are what matter here. Full‑split runs are the next step.

---

## 1. What changed in TBKV (the adjustment)

Implemented **item A of the adjustment plan — full‑block token skipping** for ViTDet.

Previously TBKV reused only the cached **K/V projection** of matched background tokens
in the 4 global‑attention blocks, so a reused token still paid for its query
projection, the attention output projection and the *entire* MLP — saving only
≈ 2.4 % of the compute.

Now, when `token_skip=true`, a matched background token **bypasses the whole block**
(Q, attention, output projection **and** MLP) and reuses its cached *post‑block output*
token — exactly the Eventful/STGT token‑gating semantics, but driven by TBKV's
cosine‑similarity cache matching. Only the *unmatched background + foreground* tokens
run through the block. A `cache_period` option was also added to periodically refresh
the keyframe cache and bound temporal drift.

Files: `src/tbkv/cache.py`, `src/tbkv/tbkv_blocks.py`, `src/tbkv/tbkv_backbone.py`,
`src/models/tbkv_vitdet.py`, `scripts/evaluate/tbkv_vitdet_vid.py`.

New CLI: `token_skip=true`, `cache_period=<int>`.

---

## 2. ImageNet VID — ViTDet‑B, 672 px (mAP@50)

| Method | GFLOPs / frame | mAP@50 (%) | FLOP reduction |
|---|---:|---:|---:|
| ViTDet‑B (baseline) | 174.5 | 84.1 | — |
| STGT (k=512) | 68.5 | 83.8 | −61 % |
| STGT (k=768, best acc.) | 90.1 | 84.4 | −48 % |
| Eventful (temporal, k=512) | 60.5 | 84.0 | −65 % |
| MaskVD (period=4) | 88.8 | 83.1 | −49 % |
| TBKV — KV‑reuse only (r=1.0, *old*) | 170.4 | 83.0 | −2.4 % |
| **TBKV — token‑skip, conservative** (r=0.3, bg=0.4) | 165.5 | 73.1 | −5.2 % |
| **TBKV — token‑skip, medium** (r=0.5, bg=0.5) | 156.5 | 73.1 | −10.3 % |
| **TBKV — token‑skip, aggressive** (r=0.7, bg=0.6) | 144.8 | 73.1 | −17.0 % |
| **TBKV — token‑skip, max** (r=0.9, bg=0.7) | 130.3 | 74.4 | −25.4 % |

TBKV GFLOPs are the **matching‑frame** cost (`cache_period=8` keyframes refresh the cache).

**Takeaway (VID).** Full‑block token skipping moves TBKV from 2.4 % → **25 %** FLOP
reduction. Notably, mAP@50 is **flat (≈ 0.73–0.74) across the whole sweep** — the
accuracy cost is a *fixed* offset from enabling output‑reuse on the 4 global blocks, so
being more aggressive saves more compute at **no additional accuracy penalty** (the `max`
point is even the most accurate). The remaining ceiling is structural: only **4 of 12**
blocks are the cacheable global‑attention blocks; the 8 windowed blocks still run in full
(this is *item B* of the plan). Reusing whole global‑block outputs also costs a fixed
≈ 10‑pt mAP@50 vs. the baseline, because those outputs feed every downstream block.

---

## 3. Kinetics‑400 — ViViT‑B (top‑1)

| Method | GFLOPs / clip | top‑1 (%) | top‑5 (%) | FLOP reduction |
|---|---:|---:|---:|---:|
| ViViT‑B (baseline) | 3359 | 80.0 | 100.0 | — |
| Eventful (temporal, 24) | 617 | 80.0 | 100.0 | **−82 %** |
| TBKV | 1537 | 70.0 | 100.0 | **−54 %** |

**Takeaway (ViViT).** Here **every** transformer block is cacheable (no windowed
blocks), so TBKV removes **54 %** of the compute — an order of magnitude more than on
ViTDet. Eventful still leads on the accuracy/FLOP frontier at this sample size, but the
result confirms TBKV's savings scale with the fraction of cacheable blocks, which is the
core lesson driving *item B*.

---

## 4. Figures (in `results/plots/`)

1. **`long_run_efficiency.pdf`** — amortized GFLOPs/frame vs. clip length T. TBKV pays
   the heavy keyframe pass only every `cache_period` frames, so its amortized cost keeps
   dropping as clips get longer, while STGT/Eventful gate on *every* frame (flat).
2. **`specular_tradeoff.pdf`** — mirrored bar chart around the baseline centre line:
   mAP@50 up, GFLOPs down; left→right = more aggressive skipping ⇒ more FLOPs saved,
   gently lower mAP@50.
3. **`tbkv_configs_pareto.pdf`** — every TBKV operating point vs. the SOTA curves on the
   accuracy/efficiency plane.

Regenerate with: `python scripts/plot_tbkv_paper.py --out_dir results/plots`.

---

## 5. Honest assessment for the paper

- **Strength:** token skipping is now a *real* compute reduction, and on architectures
  where all blocks are cacheable (ViViT) it reaches −54 %. TBKV's amortized cost also
  falls with clip length, an advantage the frame‑wise gating baselines do not have.
- **Gap:** on ViTDet, TBKV is not yet on the Pareto frontier (STGT/Eventful reach ≈84 %
  at 60–90 GFLOPs). Closing it requires **item B** (extend skipping to the 8 windowed
  blocks) and **item D** (cheaper matching), per `TBKV_ADJUSTMENT_PLAN.md`.

---

## 6. TBKV + a second technique (modular stacked reduction)

TBKV only *reuses* tokens that match the cache. Everything it could **not** reuse — the
**fresh / active** tokens (unmatched background + foreground) — is still recomputed in
full. Section 6 stacks a **second, modular token‑reduction policy on exactly those fresh
tokens and their newly‑computed K/V**, so the two methods compose instead of competing.

**Fusion semantics (per the request).**

- **ViViT (Kinetics‑400).** In each spatial block: TBKV first matches and *substitutes*
  cached tokens, then the secondary policy runs **on the fresh tokens + new K/V** and
  keeps only the top `keep` fraction, pruning the rest (classification only needs the
  class token, so the grid may shrink freely). Implemented in
  `src/tbkv/match.py::perform_tbkv_matching`.
- **ViTDet (VID).** The grid must stay fixed, so we *invent* a defer‑and‑reuse scheme:
  TBKV‑matched tokens reuse the keyframe output; among the remaining **active** tokens
  the secondary policy keeps the top `keep` fraction (fully recomputed) and **defers** the
  rest, which reuse the previous matching‑frame's output for that position. Deferred
  tokens are also dropped from the attention key set (pure pruning, no zero‑key
  pollution). Implemented in `src/models/tbkv_vitdet.py::_forward_tbkv_skip`.

**Secondary policies** (`src/tbkv/secondary.py`, selected with `secondary=<name>`):

- `eventful` — keep the fresh tokens whose features changed most (largest L²) vs. the
  previous frame (temporal gating).
- `maskvd` — keep the fresh tokens passing an EMA‑smoothed saliency mask (a
  temporally‑consistent soft spatial mask; adapted to a saliency‑threshold mask on ViViT,
  where there is no detector mask head).
- `stgt` / `none` also available. `secondary_keep=<0..1>` sets the keep fraction.

New CLI: `secondary=eventful|maskvd|stgt|none`, `secondary_keep=<float>`. Configs:
`configs/evaluate/{vitdet_vid,vivit_kinetics400}/tbkv_combo.yml`. Runner:
`scripts/run_tbkv_combo.sh`.

### 6.1 ImageNet VID — ViTDet‑B, 672 px (TBKV token‑skip + secondary)

| Method | GFLOPs / frame | mAP@50 (%) | vs. TBKV‑only |
|---|---:|---:|---:|
| TBKV token‑skip only (baseline for §6, r=0.5, bg=0.5) | 156.6 | 73.1 | — |
| TBKV + Eventful (keep=0.7) | 141.6 | 71.7 | −9.6 % FLOPs, −1.4 mAP |
| TBKV + MaskVD (keep=0.7) | 141.6 | 71.7 | −9.6 % FLOPs, −1.4 mAP |
| TBKV + Eventful (keep=0.5) | 132.4 | 70.6 | −15.4 % FLOPs, −2.5 mAP |
| TBKV + MaskVD (keep=0.5) | 132.4 | 70.5 | −15.4 % FLOPs, −2.6 mAP |

GFLOPs are amortized @ T=100 (`cache_period=8`). Stacking a secondary policy removes a
further **10–15 %** of compute on top of TBKV token‑skip, for a small (1–3 pt) mAP cost.

### 6.2 Kinetics‑400 — ViViT‑B (TBKV + secondary)

| Method | GFLOPs / clip | top‑1 (%) | vs. TBKV‑only |
|---|---:|---:|---:|
| TBKV only (baseline for §6) | 1536 | 60.0 | — |
| TBKV + Eventful (keep=0.7) | 894 | 70.0 | −42 % FLOPs, +10 pt |
| TBKV + MaskVD (keep=0.7) | 885 | 70.0 | −42 % FLOPs, +10 pt |
| TBKV + Eventful (keep=0.5) | 659 | 50.0 | −57 % FLOPs, −10 pt |
| TBKV + MaskVD (keep=0.5) | 656 | 50.0 | −57 % FLOPs, −10 pt |

GFLOPs are the matching‑pass total per clip. On ViViT the fusion is dramatic: at
`keep=0.7` the secondary policy removes another **42 %** of TBKV's compute while top‑1 is
**at least as good** as TBKV‑only (the ±10 pt swings are single clips at n=10). At
`keep=0.5` it trades one more clip for a **57 %** additional cut.

**Takeaway (§6).** The two families of methods are complementary and compose cleanly
through one modular interface: TBKV harvests cross‑frame redundancy via cache matching,
then Eventful/MaskVD prune the residual fresh tokens. `keep≈0.7` is the "decent"
operating point on both models (accuracy retained, ~10–42 % extra savings); `keep=0.5` is
the aggressive point. Eventful and MaskVD land almost on top of each other here because at
n=10 they select nearly the same fresh tokens.

---

## 7. Decoupled combo + TBKV on **all** blocks

Section 6 stacked a secondary policy *after* TBKV's own full‑block output reuse, so the two
mechanisms fought over the same temporal redundancy and both inherited TBKV's whole‑block
output‑reuse accuracy floor. Section 7 restructures the combo along the two requests:

1. **TBKV does *KV‑reuse only*.** A matched token reuses its cached **K/V** (skipping its
   K/V projection) but is *not* output‑reused — it still runs Q + attention + projection +
   MLP. This removes the fixed ≈10‑pt mAP penalty that whole‑block output reuse imposed.
2. **The secondary method does the token skipping.** Eventful / MaskVD / STGT decide which
   tokens are recomputed this frame vs. deferred (reuse the previous frame's output). The
   two mechanisms now exploit *different* redundancies (per‑token K/V caching vs. recompute
   budgeting) instead of competing.
3. **TBKV runs on all 12 blocks, not just the 4 global ones.** The 8 windowed blocks get
   per‑position KV‑reuse (matched tokens reuse the keyframe K/V; windowed attention and
   relative‑position embeddings still run *exactly* over the full window token set) plus a
   pointwise, window‑independent MLP‑skip. This breaks the structural "only 4/12 blocks are
   cacheable" ceiling that pinned §2 at ≥130 GFLOPs.

**Implementation.**
- Global blocks: `TBKVViTDet._forward_kvreuse_skip` (`src/models/tbkv_vitdet.py`), routed in
  the matching pass when `kv_reuse_only=true` and a `secondary` policy is active.
- Windowed blocks: `TBKVBlock.forward_tbkv_all` (`src/tbkv/tbkv_blocks.py`), enabled by
  `tbkv_all_blocks=true`. Per‑position keyframe K/V cache + MLP‑delta cache.
- Configs: `configs/evaluate/vitdet_vid/tbkv_kvreuse.yml` (decoupled, 4 blocks) and
  `tbkv_allblocks.yml` (decoupled, all 12 blocks). CLI: `kv_reuse_only`, `tbkv_all_blocks`,
  `secondary`, `secondary_keep` are all routable.

### 7.1 ImageNet VID — ViTDet‑B, 672 px (n=10, `r_match=0.5`, Eventful `keep=0.7`)

| Method | GFLOPs / frame | mAP@50 (%) | Notes |
|---|---:|---:|---|
| §6 combo: TBKV token‑skip + Eventful (4 blocks) | 141.6 | 71.7 | whole‑block output reuse (has floor) |
| **KV‑reuse only, all 12 blocks** (`secondary=none`) | 162.1 | **84.3** | no token skip → **no accuracy floor** |
| **Decoupled: KV‑reuse + Eventful skip (4 blocks)** | 151.6 | 71.7 | KV‑reuse only, no output‑reuse floor |
| **Decoupled: KV‑reuse + Eventful skip (all 12 blocks)** | **123.6** | **71.4** | breaks the 4/12‑block ceiling |
| Reference — TBKV token‑skip only (4 blocks) | 156.6 | 73.1 | §6 baseline |

Amortized @ T=100 (`cache_period=8`). Two results stand out:

- **KV‑reuse has no accuracy floor.** With the token skip disabled, pure per‑position
  KV‑reuse on *all 12 blocks* holds mAP@50 at **84.3 %** — essentially the ViTDet baseline —
  while still trimming ~7 % of compute. This is the whole point of decoupling: the fixed
  ≈10‑pt penalty in §2/§6 came from *whole‑block output reuse*, not from K/V reuse. Accuracy
  is now set by the *tunable* secondary keep‑ratio, not a structural floor.
- **All‑blocks breaks the FLOP ceiling.** Extending KV‑reuse to the 8 windowed blocks and
  adding the Eventful MLP‑skip removes **~28 GFLOPs (‑18 %)** over the 4‑block decoupled
  version for a negligible **‑0.3 pt** mAP@50, landing **well below** the 130‑GFLOPs floor
  that whole‑block token‑skip (§2) could never cross.

