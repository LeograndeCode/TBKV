# AAAI-style review of the TempoMem draft, and what was changed in response

> Naming: the paper's method name is **TempoMem**. "TBKV"/"eventful_tbkv"
> survives only as the legacy name in the code and configs — same method.

Part 1 is written as an AAAI reviewer would write it (about the draft as it
stood before assembly). Part 2 lists the changes already applied in
`tempomem_aaai27.tex`. Part 3 is the remaining to-do list that only experiments
or authorial decisions can close.

---

## Part 1 — Simulated AAAI review (of the original draft)

**Summary.** The paper proposes TBKV, a training-free framework that caches
merged "prototype" tokens and their K/V projections from a warm-up window
and, on later frames, reuses cached K/V for matched background tokens while
substituting them into the sequence. It runs standalone (attention-entropy
foreground/background split) or as a filter on Eventful Transformers.
Results: strong on class-token classification (ViViT, VideoMAE), an
honestly reported negative result on dense detection (ViTDet/VID).

**Strengths.**
1. The core question ("which computations should never be recomputed?") is
   a genuinely different framing from per-frame token selection, and the
   content-based vs. position-based reuse distinction is well argued.
2. The Eventful integration with γ=0 reducing *exactly* to Eventful is an
   unusually clean controlled-comparison design.
3. The negative detection result is reported rather than hidden, and the
   readout-based analysis (global vs. dense) is plausible and supported by
   two independent observations (granularity sensitivity, standalone
   pattern).
4. Break-even analysis (M < 2rC) makes a testable prediction that the
   backbone-scaling experiment confirms.

**Major weaknesses (blocking as submitted).**
1. **The paper has two names for one method.** Title/abstract/intro/
   conclusion say "TBKV / Persistent Scene Memory / Persistent Scene
   Objects"; methodology/evaluation say "TempoMem / prototype cache". A
   reviewer cannot tell if this is one method or two. *(fixed — see Part 2)*
2. **No Related Work section.** AAAI expects one; the intro's inline survey
   is not citable coverage of DynamicViT/EViT/A-ViT/ATS/MeMViT etc.
   *(fixed)*
3. **Headline numbers are on tiny subsets** (100 K400 clips → ±9-point
   Wilson CI; 25 VID videos) without the paper saying so. Claims like
   "preserves Top-1 exactly" are not supported at this sample size.
   *(disclosed now; full-eval must replace — Part 3)*
4. **The VideoMAE scaling numbers are not believable as stated**: 90.5–94.0
   Top-1 on Kinetics-400 exceeds published VideoMAE full-val results
   (ViT-H ≈ 86.6). Either the eval is a subset/nonstandard protocol or the
   dataset label is wrong. Reviewers will check this against the VideoMAE
   paper in minutes. *(flagged in-file; must be resolved — Part 3)*
5. **Internal inconsistencies a reviewer will catch:**
   - Table 1 dense ViViT-B = 537.6 GFLOPs/clip @ 71.0 Top-1, but Table 3's
     caption says dense = 3355 GFLOPs/clip @ 73.0 Top-1. Both are "dense
     ViViT-B on K400". The protocols differ (matching-only vs whole-clip),
     but the paper never reconciles the *dense* rows across tables.
   - Table 1: Top-1 preserved exactly (71.0) while Top-5 falls 10 points
     (97→87). One sentence of explanation exists but this pattern is odd
     enough to need either error bars or a short analysis (e.g., logit-tail
     perturbation histogram).
   - Abstract/intro of the original draft promised "benefits grow with
     video duration" and cumulative-FLOPs-vs-length experiments; the
     evaluation contains no such experiment. *(claims removed/aligned)*
6. **No method overview figure.** For a mechanism with four moving parts
   (warm-up, merge, match+substitute, refresh), a diagram is not optional
   at AAAI. *(placeholder added — must be drawn)*
7. **Latency is worse than dense** for standalone TBKV (now TempoMem) (101.8 ms vs 69.6
   ms). The paper is honest about it, but then the contribution list must
   not imply deployment-readiness; "FLOPs reduction" claims should always
   be qualified. *(contributions rewritten accordingly)*

**Minor weaknesses.**
- Preamble bugs: `algorithmic` and `algpseudocode` both loaded (clash),
  `booktabs` loaded twice, `\newtheorem{...}[section]` with
  `secnumdepth=0` breaks numbering, `\ref{sec:...}` prints nothing in this
  template. *(all fixed)*
- The old generic Algorithm 1 (`alg:recompute`) was superseded by the two
  algorithms in `algorithms.tex` but the prose still referenced it.
  *(replaced per the placement notes in algorithms.tex)*
- Fig. 1 (ViViT curve) legend uses "ρ" for the reuse fraction while the
  text uses γ (the TempoMem name in the legend is correct). *(must
  regenerate — Part 3)*
- W and P (warm-up length, refresh period) are introduced as the two main
  knobs after r, then never swept. The paper admits this ([TBD]), but at
  least a small P sweep is needed for the failure-mode claims to stand.
- Saliency score Eq. (negative entropy): sign/normalization convention is
  stated loosely ("normalized to [0,1]"); make min-max explicit.
- Contribution 5 of the original draft was "an evaluation plan" — a plan is
  not a contribution. *(removed)*

**Questions to the authors.**
1. Why does Top-5 degrade 10 points when Top-1 is unchanged (Table 1)?
2. How was the 100-clip/25-video subset chosen, and are all methods paired
   on identical clips? (Now stated: yes — keep it stated.)
3. What happens at an actual scene cut between refreshes? The failure-mode
   analysis predicts degradation bounded by P; show one stress test.
4. Is the ViTDet KV-reuse-only variant's 156.5 GFLOPs (barely below dense
   174.5) worth reporting as a method, or should it be framed purely as an
   ablation?

**Rating (as submitted): 4/10 (borderline reject).** Interesting idea,
clean experimental design, honest reporting — but preliminary-scale
evidence, unverifiable scaling numbers, and missing related work.
**With full-eval numbers, the figure fixes, and the scaling-table
verification, this plausibly becomes a 6 (weak accept), leaning on the
strength of the controlled Eventful comparison and the readout analysis.**

---

## Part 2 — Changes applied in `tempomem_aaai27.tex`

1. **Unified naming**: TempoMem everywhere, including the title
   ("TempoMem: Persistent Scene Memory for Efficient Video Transformer
   Inference"). "Persistent Scene Memory (PSM)" is kept, defined once in
   the intro as the name of the per-block cache and linked explicitly in
   the Methodology ("this per-block cache is the PSM of the
   Introduction"). "PSO" and "TBKV" are gone from the paper (TBKV remains
   only as the code/config name); the memory's units are "prototypes"
   throughout.
2. **New Related Work section** with three paragraphs mirroring the
   intro's taxonomy: spatial token reduction (ToMe, EViT, DynamicViT,
   A-ViT, ATS), temporal redundancy (Eventful, STGT, DeltaCNN, MaskVD),
   and memory-augmented video transformers (MeMViT, TTM) including the
   LLM KV-cache analogy and a clear statement of what TempoMem does
   differently in each case.
3. **Rewritten abstract**: mechanism-first, states the actual measured
   results (27% / 86% / 3.8×), and explicitly includes the negative
   detection finding with its explanation — this converts a weakness into
   a credibility signal.
4. **Rewritten introduction**, following the original draft's framing per
   your instruction (per-frame-decision critique, persistence argument,
   "which computations should never be recomputed", the two complementary
   benefits, amortization) but tightened, aligned with the actual results
   (no cumulative-FLOPs or "grows with duration" claims that the
   evaluation doesn't support), and ending in 4 evidence-backed
   contributions instead of 5 (the "evaluation plan" item is gone; the
   readout-scoping finding is promoted to a contribution).
5. **Algorithms**: the two final algorithms from `algorithms.tex`
   (standalone; Eventful filter) replace the old generic one; all prose
   references updated per the placement notes.
6. **Evaluation-scale paragraph added** (100 clips / 25 videos, paired
   subsets, ±9-point Wilson CI, full-eval [TBD]) so no reviewer discovers
   the scale on their own.
7. **Real figures wired in**: `fig_vivit_k400.png` (with corrected caption
   matching what the plot actually shows: Eventful budget sweep vs TempoMem γ
   sweep, per-clip FLOPs, CIs) and `fig_pareto_vid.png` as an honest
   frontier figure for the negative result.
8. **Preamble fixed**: dropped `algorithmic`, deduped `booktabs`,
   `\newtheorem` without `[section]`, `\graphicspath`, no `\ref{sec:...}`
   anywhere.
9. **Bibliography created** (`references.bib`, 17 entries) and
   `\bibliography{references}` added.
10. **Conclusion rewritten** to match the evidence and to end on three
    concrete future directions (adaptive budgets, detection-aware reuse,
    kernel fusion) instead of restating the intro.
11. **In-file warnings** added at the two danger points: the VideoMAE
    scaling table (numbers exceed published SOTA — must verify) and the
    ViViT figure (ρ→γ legend fix).

## Part 2b — Results audit (2026-07-17): unbacked claims removed

Every number in the paper was checked against `results/`. Removed because
the repo does not back them (restore only if/when real runs produce them):

- **Standalone ViViT table** (537.6→392.0 GFLOPs, "71.0 Top-1 preserved",
  +ToMe 155.4/63.7). The repo's actual standalone runs
  (`results/evaluate/vivit_kinetics400/tbkv-n_items=100`) score **54.0/48.0
  Top-1 vs dense 73.0** — the claim was contradicted, not just unverified.
  The paper now has a "Standalone TempoMem: Work in Progress" subsection
  stating the honest 54.0 number and that retuning + full-scale runs are
  [TBD].
- **VideoMAE S/B/L/H scaling table and UCF101 result**. No such runs exist;
  the only VideoMAE artifact is a ViT-B port with broken head weights
  (Top-1 0.04–0.12 in its metrics.json). All scaling/UCF claims deleted
  from abstract, intro, contributions, setup, metrics, summary, conclusion.
- "Break-even prediction confirmed by experiments" → reworded as a design
  property, since the confirming experiment (scaling table) is gone.

Added because the repo *does* back it but the paper didn't use it:

- **Substitution on/off ablation** (`results/comparison/substitute_test.txt`):
  ViViT γ=0.5 identical 59.0/79.0 with substitution on or off (288 vs 222
  matching GFLOPs/clip); ViTDet γ=0.25 substitution 87.2 vs 86.9 mAP@50.
- **Full-val Eventful anchors** (19,877 clips: k=24 → 62.4 Top-1 @ 27.2
  GF/frame; k=48 → 67.5 @ 53.8) as a subset-bias check in the setup.
- The VID SOTA table row was renamed "TempoMem (KV-reuse only)" to match
  the variant actually run on ViTDet, and the latency/memory numbers were
  verified against the validated n=10 harness in
  `scripts/plot_paper_eval.py`.

Also in this pass: all "---" em-dashes removed from the body text.

Later reframing (same day, per author): the ViTDet finding is no longer
called a "negative result". The data supports "a much less favorable
accuracy-compute trade-off" (mAP@50 falls in proportion to the reuse
rate); the stronger "does not improve on Eventful's frontier" claim is
kept but explicitly scoped to the paired 25-video sweep, with full-dataset
confirmation pending the γ=0 full-VID run.

## Part 3 — Remaining before submission (needs you / experiments)

- [ ] Full-validation runs replace all `%% PRELIM` numbers (19,877 K400
      clips; 639 VID videos). Search for `[TBD]` and `PRELIM`.
      Progress 2026-07-17: full-VID Eventful+TempoMem γ=0.5/k=128 done
      (60.5 mAP@50 @ 12.4 GF/frame, now in the paper); the paired γ=0
      full run is executing (`results/comparison/vitdet_full_cr0.log`) —
      when it lands, add both to a full-dataset table and re-verify the
      Pareto conclusion at full scale.
- [ ] If you want the backbone-scaling story back: obtain proper VideoMAE
      full checkpoints (current port has head-only weights, Top-1 ~0.04),
      run S/B/L/H on K400 (and optionally UCF101), then reinstate a
      scaling table with the measured numbers.
- [ ] Retune the standalone saliency split (β, refresh) and re-run
      standalone ViViT at full scale; the paper currently reports the
      honest 54.0 vs 73.0 subset result and promises [TBD].
- [ ] Regenerate `fig_vivit_k400.png` with the γ symbol in the legend, and
      the VID pareto with a "TempoMem" legend (currently "TBKV (ours)")
      plus the Eventful+TempoMem whole-stream point once measured.
- [ ] Draw the pipeline overview figure (placeholder comment in the
      Pipeline subsection).
- [ ] W and P sweeps + one scene-cut stress test (currently [TBD]).
- [ ] Whole-stream row for Eventful+TempoMem in the SOTA table ([TBD]).
- [ ] Fill the AAAI reproducibility checklist; decide on code release
      statement (anonymized repo link).
- [ ] Get the AAAI-27 author kit (`aaai2027.sty`/`.bst`) — not in this
      repo; the paper will not compile without it. Check page budget after
      figures land (AAAI-26 was 7 pages + references; verify AAAI-27).
