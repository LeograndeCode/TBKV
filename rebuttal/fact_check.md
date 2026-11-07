# Fact check of the four responses (26 Oct 2026)

Sources:
- **Paper:** the PDF in the repo root (`38867_Persistent_Semantic_Memo.pdf`), checked page by page.
- **Code:** `submission/` (the code of record) and its stored results.

Every claim below is marked **✓** (verified) or **≈** (verified but approximate, with a note).

## Claims and sources

| Claim (responses) | Source | Status |
|---|---|---|
| τ appears on p. 3 (R1, R2, R3, R4) | Paper p. 3: Memory Construction text and Eq. 2 | ✓ |
| τ is "a leftover from an earlier version" | Author's statement; `src/evit/blocks.py` on `main` has `reuse_threshold` | ✓ |
| What runs is Algorithm 1's reuse fraction γ | Paper p. 4 (Alg. 1, l. 7); code `src/psm/filter.py:36` | ✓ |
| Skip count is "γk" | Code skips `round(γk)` (128 at k=512, γ=0.25; 23 at k=24, γ=0.95). Alg. 1 writes ⌊γk⌋, so the responses avoid floor | ✓ |
| Memory = tokens the host recomputes in the first W=4 frames | Code `blocks.py:151` (caching appends gate output); defaults `warmup=4` / `frame_split=4`; paper p. 4 "W = 4" | ✓ |
| Merge update d ← (d + Σa)/(1 + n) | Code `merge.py:75`, `scatter_reduce(..., "mean")` including d itself; one pass halves the set | ✓ |
| 52 prototypes per block at 672, ~2 MB in all | Ran `compute_merge` on 3,300 tokens (1764 + 3·512), T=6 → 52; 52 × 768 × 4 B × 12 blocks = 1.92 MB | ✓ |
| Rebuilt every N frames, unused in our runs | Code: `cache_period` option in the VID script, off in every reported config | ✓ |
| "~10 s clips" | K400: paper p. 7 "clips are ten seconds". VID: not stated in the paper; average ≈255 frames per video from the stored counts | ≈ |
| γ=0.25, T=6 (detection); 0.95, 2 (K400), set on a 10% subset | Paper p. 4, Ablation split paragraph | ✓ |
| Skipped tokens keep the host's stored K/V | Paper p. 4, Alg. 1 l. 14; code (forced index, host `TokenBuffer`) | ✓ |
| 49% → 39%, 2.5 → 2.6 | Paper p. 2 (49%), pp. 1–2 (2.5); Table 3 p. 7: 62.38 → 59.82 = 2.56; 1053 → 645 = 38.7% | ✓ |
| *Internal only, in no response:* K400 issue, class token not recomputed when γ>0 | Code audit plus `check_frozen_readout.py`, which gives bit-identical outputs for any γ>0 | ✓ (random weights; re-run with the trained checkpoint) |
| *Internal only, in no response:* PSM 58–60% at every budget | Table 3 p. 7: 59.82 / 58.37 / 58.43 | ✓ |
| γ=0 is the host | Paper p. 5 "At γ = 0 the pipeline reduces to Eventful itself" | ✓ |
| Of 2,329 MB, 2,324 MB is the host's | Table 1 p. 6 (PSM 2329, Eventful 2324); paper p. 5 | ✓ |
| Faster than dense (60.0 vs 70.1 ms), near parity with host | Table 1 p. 6 (host 58.9); paper p. 5 "near parity" | ✓ |
| *Internal only:* at 672 PSM beats Eventful + pooling (79.87 vs 79.48); lowest at 1024 | Table 1 p. 6 (1024: 82.93 / 79.42 / 76.98 / 76.70 / 75.57) | ✓ |
| Accuracy falls more at 1024 and at high γ | Table 1 (−1.93 at 672 vs −3.85 at 1024); Fig. 2 / p. 6 | ✓ |
| Budget-matched host recovers ~1.5 mAP (p. 5) | Paper p. 5; measured Eventful k=384 = 81.40 in `results/.../temporal_672/output.txt` | ✓ |
| T moves mAP ≤1.2 for γ ≤ 0.75 (p. 6) | Paper p. 6; 64-video grid ranges 0.60 / 0.67 / 1.12 (2.80 at γ=0.95, hence the qualifier) | ✓ |
| Replay models archived footage (pp. 4–5) | Paper: replay paragraph starts on p. 4, "archived or looped footage" is on p. 5 | ✓ |
| Latency likely limited by unfused matching (p. 5) | Paper p. 5 says this attribution is "unverified", hence "likely" | ✓ |
| *Internal only:* matching cost is not in the FLOP count (the revision should report it) | Counters unchanged across 192 `psm_keep_mask` calls; identical counts to Eventful at k=18 | ✓ |
| Appendix plots possible from existing runs | T grid (`submission/results`, 64 videos); host six-budget sweep (`temporal_672`); PSM full-split points (672, 1024). **No full-split PSM γ sweep exists**, so the responses promise "the host's sweep with PSM" | ✓ |
| VID / K400 follow the host's and baselines' protocols | Paper p. 4 ("following Dutson et al.; Sarkar et al."), p. 6 (ViViT in the Eventful configuration) | ✓ |
| Deep Feature Flow, MeMViT, TTM descriptions | Paper p. 2 related work (MeMViT, TTM); Deep Feature Flow's standard description (trained flow network) | ✓ |
| ProtoKV, VLMCache, StateKV, STTM | ProtoKV, VLCache (the reviewer writes "VLMCache") and STTM verified online; StateKV not found, so only the reviewer's general framing is used | ≈ |


### Reviewer 1, point 3 (revised wording)

| Claim | Source | Status |
|---|---|---|
| Drop-in layer, no retraining | Paper pp. 2, 4 (training-free, frozen checkpoints, host untouched) | ✓ |
| γ=0 is exactly the host | Paper p. 5; code README ("bit-for-bit the unmodified host") | ✓ |
| 60.7 → 46.6 GFLOPs/frame, lowest in Table 1 (672 block) | Table 1 p. 6 (others: 174.5 / 60.7 / 52.8 / 68.5 / 80.9). Still the lowest after adding the ~0.25 GFLOPs of uncounted matching | ✓ |
| Costs 1.9 mAP at 672, 3.9 at 1024 | Table 1: 81.80 → 79.87 (1.93); 79.42 → 75.57 (3.85) | ✓ |
| ~5 MB fixed memory; the other 2,324 MB is the host's | Paper p. 5 ("does not grow with video length") | ✓ |
| Suits recurring content such as stationary cameras (pp. 4–5) | Paper pp. 4–5: replay "models deployments in which content recurs ... (archived or looped footage, stationary cameras)". This is the paper's own framing, not a new claim | ✓ |
| 14% faster than dense, near parity with the host | Table 1: 60.0 vs 70.1 ms (−14.4%); host 58.9 ms | ✓ |

The Reviewer 1 response no longer mentions the Kinetics-400 issue (author's decision). It has since been removed from all four responses at the author's request; disclosure via the confidential comment to the SPC/AC is recommended. Reviewer 1's point 2 only promises to align the numbers across sections; it does not present the K400 figures as validated.


### Reviewer 2 (revised wording)

| Claim | Source | Status |
|---|---|---|
| VID and K400 are the standard benchmarks for comparing these accelerators | Paper p. 4 (follows Dutson et al.; Sarkar et al.), p. 6 (ViViT in the Eventful configuration) | ✓ |
| "Videos of a few hundred frames" | Not stated in the paper. Estimated ≈230–290 frames/video from the stored counters; verify with `labels.json` | ≈ |
| One memory from the first frames kept the full saving | Full-split replay run: 46.6 GFLOPs on every matching frame (deterministic with top-k) | ✓ |
| Per-frame saving is a fixed share of the host budget, independent of length | Code: `n_keep = k - round(γk)` on every frame | ✓ |
| With rebuilds, a long video is a chain of segments, "so we expect results to hold" | Design expectation (the rebuild never triggers in our runs), worded as an expectation | ✓ |
| 60.7 → 46.6 GFLOPs/frame | Table 1 p. 6 | ✓ |
| W4: PSM 14% below dense; the host turns 65% fewer FLOPs into 16% lower latency | Table 1 p. 6 at 672: 1 − 60.0/70.1 = 14.4%; 1 − 60.7/174.5 = 65.2%; 1 − 58.9/70.1 = 16.0% | ✓ |
| W4: larger K400 losses at higher budgets | Table 3 p. 7: gap to Eventful 2.56 / 9.17 / 17.28 points at k = 24 / 48 / 96. No cause is given | ✓ |

## Fixed in this pass

- **Page reference:** "archived footage (p. 4)" became **(pp. 4–5)**.
- **Skip count:** "⌊γk⌋" became **"γk"**. The code rounds; floor would differ for K400.
- **Memory size:** "~2 MB" became **"~2 MB in all"**. It is about 160 KB per block.
- **Latency:** "on par with the host" became **"near parity"** (the paper's own words; PSM is 1.1 ms slower).
- **T ablation:** "T moves mAP ≤1.2" now says **"for γ ≤ 0.75"**. At γ = 0.95 the range is 2.8.
- **Undefined symbols:** |B| and S_d were removed or defined in words.
- **Reviewer 1:** added the γ/T values and the tuning procedure; the "lower accuracy" item; 49 → 39 stated explicitly.
- **Reviewer 3:** Q1 opens with "not yet"; Q5 admits other baselines lack sweeps; Q8 moved after Q7; the latency attribution is hedged as "likely".
- **Reviewer 2:** the K400 class-token sentence is gone (author's decision). W4 now gives the host's FLOPs-to-latency ratio and acknowledges the larger K400 losses at higher budgets without naming a cause. W2 ends with combining memory and change signals as the next step.
- **Reviewer 4:**
  - "one memory per block" is now explicit;
  - the 1024 / high-γ accuracy drop and optimized kernels are acknowledged;
  - MeMViT and TTM are now described as "trained with a memory", not "learn a memory", because MeMViT caches activations.

## Still to do before posting

- Run `check_frozen_readout.py` with the trained K400 checkpoint before writing the confidential comment or the camera-ready. No response depends on it.
- Paste each response into its box and confirm the counter shows ≤1,500. Without the final newline the texts are 1,492 / 1,491 / 1,495 / 1,487 characters. If the box counted UTF-8 bytes instead, R1 and R3 would be 1,504 and 1,508, because γ, Σ and the arrows take 2–3 bytes each.


## AI review and confidential comment (27 Oct 2026)

Paper text extracted from the PDF with a small stream parser (no PDF library on this machine).

| Claim | Source | Status |
|---|---|---|
| The only matched-budget control is p. 5's interpolated host point, ~1.5 mAP better | Paper p. 5: "a budget-matched Eventful configuration interpolated from its published sweep would recover roughly 1.5 mAP" | ✓ |
| No random-skip control was run | The paper never mentions "random"; no such run in `submission/results` or the scripts | ✓ |
| PSM's own share: 60.7 → 46.6 GFLOPs/frame, 8 of the 73 points vs dense | Table 1: 1 − 60.7/174.5 = 65.2%; 1 − 46.6/174.5 = 73.3%; difference 8.1 | ✓ |
| "Three quarters" reading | Fig. 2 discussion: γ = 0.25 "already delivers almost three quarters of the achievable saving" (dense-relative, so it includes the host's saving) | ✓ |
| Both protocols test reuse within short clips after a bootstrap | Paper p. 4: detection bootstrap is the first W = 4 frames, "unscored, uncounted"; K400 caches 4 temporal steps and matches 12 | ✓ |
| Savings are linear in γk on the implemented path | Alg. 1 l. 14 (skipped tokens take the host's reuse path); counters: PSM linear+matmul 45.18 G vs Eventful k=384 45.66 G | ✓ |
| Eqs. 3–4 mix FLOP conventions | Eq. 3: 4NC² + 8NC² (one op per MAC) + 4N²C (two per MAC) | ✓ |
| Table 1 uses op counters, not the equations | Paper p. 4 ("in-code operation counters"); dense linear count 1.498e11 = 12 blocks × 1764 × 12C² (one op per MAC) | ✓ |
| Counters omit matching, ~0.25 GFLOPs/frame | Earlier audit (counts unchanged across `psm_keep_mask` calls); 512 × 52 × 768 × 12 = 0.245 G MACs. Contradicts p. 4's "includes the cost of PSM's own matching" (known error, author notes) | ✓ |
| Near host latency, behind the other gated methods | Table 1 at 672: 60.0 vs 58.9 ms; STGT 55.1, MaskVD 54.9, pooling 49.0; p. 5 "slightly behind the other gated baselines" | ✓ |
| At 1024 spatial pooling is faster and more accurate | Table 1: 98.5 vs 143.0 ms; 76.98 vs 75.57 mAP. Not "dominates": PSM is 1.8 GFLOPs cheaper | ✓ |
| Minor: query wording | Paper p. 3 "produced by the current transformer block"; code matches on the LayerNorm-ed block input (`src/psm/blocks.py`, `_acc`) | ✓ |
| Minor: frozen memory, subsample overlap | Alg. 1 l. 3 merges once on the last caching frame vs "constructed online" (p. 3); p. 4 "subsamples come from the validation splits" | ✓ |
| *Confidential:* matched-budget gap, 1024 shortfall and failure regimes are reported by the paper itself | pp. 5–6; Fig. 2; the AI review's fourth strength | ✓ |
| *Confidential:* class token not updated during matching when γ > 0; detection unaffected | Code audit plus `check_frozen_readout.py` (random weights). Consistent with the paper's own numbers: Table 2 flat within 0.1 Top-1 across γ, Table 3 PSM at 58–60% at every k. ViTDet has no class token | Removed from the comment at the author's request (27 Oct 2026) |

Lengths at drafting: AI review 1,486 characters; confidential comment 1,062 (now 815, see the final check).


## Final check before posting (27 Oct 2026)

All six texts were checked for length, links, identifying details, banned content (K400 cause, "appendix", new experiments, "23%", "injected"), consistency between `.md` and `.tex`, and LaTeX escaping. Changes made:

- **R3 Q2, R4 W3:** removed "(~2 MB total)". The paper (p. 5) says "the prototype cache adds only about 5 MB", and R1 says ~5 MB. The 2 MB figure (prototype tensors only) read as a contradiction.
- **R1 point 2:** "we will correct 49% to 39% and 2.5 to 2.6 points" became "we will make these numbers consistent throughout". The 2.6-point figure is a K400 PSM accuracy, which the confidential comment says does not measure the method. Restating it would break the honesty floor and conflict with that comment.
- **AI review, minor:** "49% → 39%" became "the 49% figure", for the same reason.
- **R4 W2:** "modelling" → "modeling" (US spelling, as elsewhere).
- **LaTeX headers:** refreshed the stale character counts.

Final lengths (characters / UTF-8 bytes): R1 1,479 / 1,491; R2 1,491 / 1,495; R3 1,471 / 1,484; R4 1,456 / 1,462; AI 1,491 / 1,498; confidential 815 / 817. All are within 1,500 under either count.
