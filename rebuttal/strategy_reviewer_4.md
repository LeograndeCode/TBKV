# Rebuttal strategy — Reviewer 4 (score 5, confidence 4)

**Rules:** at most 1,500 characters, no links, deadline 31 Oct 2026 (AoE). The research behind this strategy is in `strategy_reviewer_1.md` §1.

## Reviewer profile

- Score 5: marginally below the threshold, so this reviewer is persuadable.
- Specific points for rebuttal: "main concerns in the weaknesses", i.e. W1–W4.
- Strengths they credit: clear problem formulation and implementation; broad empirical coverage.
- Their stated reasons for weak reject:
  - novelty not distinguished from prior memory and feature-reuse work;
  - replay-based evaluation;
  - incomplete details;
  - brittle trade-offs.
- **The only point unique to this reviewer is W1 (novelty).** W2–W4 overlap with Reviewers 1–3, so the answers must use the same wording.

## Part 3. Classification

| Weakness | Category | Real concern | Resolution | Convincing evidence | Impact |
|---|---|---|---|---|---|
| W1 Novelty vs Deep Feature Flow, MeMViT, TTM | Possible misunderstanding + positioning | Is PSM just feature reuse or memory under a new name? | Clarification: three-way contrast + a committed comparison table | What each method caches, how it indexes it, whether it trains, what it is for | **Highest for this reviewer**, and answerable |
| W2 Online streaming not tested | Missing evidence; claim exceeds evidence | Replay, frozen memory, no drift test | Narrow the claim + limitation | p. 4 already says replay models archived footage; the periodic rebuild exists but isn't exercised on ~10 s clips | Medium |
| W3 Implementation details (capacity, eviction, threshold, layers) | Missing detail / reproducibility | "Bounded memory" claimed but not specified | Clarification | Capacity formula and value, rebuild instead of eviction, no τ, one memory per block, update equation | Medium; easy win |
| W4 Brittle trade-offs | Unfavourable result | Latency flat, accuracy drops at 1024 / high γ | Context + concession + narrowed claim | Table 1 latency vs dense and host; p. 5 budget-matched point; Kinetics-400 issue disclosed | Medium-high |

## Part 4. Strategy

- **A. Approach.**
  - Lead with W1. It's this reviewer's distinctive concern, and it has the best available answer.
  - Then W3 (an easy win), then W2 and W4, conceding with narrowed claims.
- **B. Priority.** W1 > W3 > W4 > W2, ranked by ability to change this reviewer's mind.
- **C. Evidence.** All from the paper (Related Work, p. 4 replay, p. 5 budget-matched, Table 1), plus favourable implementation facts (capacity, rebuild). No new numbers.
- **D. Central concern (W4).** Same as for Reviewers 1 and 2:
  - put latency in context;
  - concede the budget-matched point;
  - plot from sweeps that already exist;
  - narrow the claim.

  Don't promise a fused kernel's speed-up.
- **E. Experiments.** None in the rebuttal. A numerical comparison with Deep Feature Flow would need a CNN detector and a trained flow network, outside the training-free ViT setting. Say so if space allows; otherwise give the comparison table only.
- **F. Style.**
  - Describe the three prior methods neutrally and accurately; no disparaging comparisons.
  - Cite Reviewer 3's agreement on the "memory for computation, not context" distinction.

## Part 5. Checklist

- [ ] W1: Deep Feature Flow / MeMViT / TTM contrast; Deep Feature Flow added; comparison table committed
- [ ] W2: replay = archived footage; rebuild not exercised on ~10 s clips; drift untested; claim narrowed
- [ ] W3: capacity, bounded memory, rebuild (no eviction), no τ, update equation, one memory per block
- [ ] W4: latency in context; budget-matched concession; sweeps plotted; Kinetics-400 issue disclosed
- [ ] ≤1,500 characters, no links, wording consistent with Reviewers 1–3
