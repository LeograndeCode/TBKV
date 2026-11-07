# Rebuttal strategy — Reviewer 3 (score 6, confidence 5): the champion

**Rules:** at most 1,500 characters, no links, deadline 31 Oct 2026 (AoE). The research behind this strategy is in `strategy_reviewer_1.md` §1.

## Reviewer profile

- This is the only positive score, from the most confident reviewer.
- [Evidence, Gao et al. 2019] Final scores tend to move toward the other reviewers' scores. The realistic goal is therefore to give this reviewer what they need to **defend the paper in discussion**: precise answers to all eight questions, and no claim that could later prove wrong.
- Losing this reviewer's trust is the biggest risk in the whole rebuttal.
- One misconception in their summary to correct gently: they believe matched tokens "reuse cached representations or key/value projections" *from the prototypes*. In fact, skipped tokens keep the host's own stored K/V.
- Their Strength 4 (recognition tolerates aggressive reuse) rests on Kinetics-400 results affected by the class-token issue. **They must hear about it from us.**

## Part 3. Classification of Q1–Q8

| Q | Category | Answer type | What convinces | Impact |
|---|---|---|---|---|
| Q1 single-pass, all costs counted | Missing evidence; claim exceeds evidence | Limitation + narrow the claim (no new results) | p. 4 already says replay models archived footage; single-pass memory building = the host's first frames + one merge | High (their main reservation) |
| Q2 capacity, replacement, τ, update equations | Missing detail / reproducibility | Clarification | Exact rule, capacity formula and example, no τ, mean update, one memory per block | High, and fully answerable |
| Q3 recurrence gap | Missing evidence | Limitation | Benchmarks have ~10 s clips | Medium |
| Q4 false-match rate | Missing evidence | Limitation + indirect evidence | p. 5: a budget-matched host recovers ≈1.5 mAP → some matches harm detection | Medium |
| Q5 Pareto curves | Missing evidence; unfavourable result | Commit to plotting existing sweeps; be honest about missing ones | The host's budget sweep and PSM's γ sweep exist; other baselines' sweeps don't | Medium-high |
| Q6 latency vs FLOPs | Unfavourable result | Explanation | Unfused matching per block; batch-1 overhead; the host shows the same gap; matching FLOPs to be added to the count | Medium |
| Q7 stability of γ, T | Missing evidence | Partial answer + limitation | T ≤1.2 points (p. 6); γ per task, chosen on a fixed subsample; Kinetics-400 issue disclosed | Medium |
| Q8 permanent scene change | Missing analysis | Mechanism + limitation | Periodic rebuild bounds staleness to one period; a similarity floor is future work | Medium |
| W6 related work | Positioning | Clarification | Those methods target the LLM side (KV cache or token reduction in VLM/video-LLM decoding); PSM targets per-frame recomputation in the vision backbone | Low, but cheap |

## Part 4. Strategy

- **A. Approach.** Answer Q1–Q8 in order, one line each where possible, because this reviewer will check them off. Thank them briefly; they gave substantive praise.
- **B. Priority.** Q2 and Q1 first: they are the most answerable and the most important. Then Q5–Q8. Related work last, in one sentence.
- **C. Evidence.** Everything comes from the paper (p. 4 replay statement, p. 5 budget-matched, p. 6 T) plus favourable implementation facts (capacity, rebuild). No new numbers beyond the paper except the prototype count and size, which describe the method rather than report a result.
- **D. Central concern (Q1/Q5).** Narrow the streaming claim, which the paper already partly does on p. 4. Commit only to plots built from runs that already exist.
- **E. Experiments.** None for the rebuttal. Camera-ready candidates: single-pass/whole-stream detection, a gap/scene-cut protocol, a match-level error analysis, Kinetics-400 with the fix.
- **F. Style.**
  - Precise and technical (confidence-5 reviewer).
  - Say "not measured" plainly. Don't hedge.
  - Get the related-work technical facts right: no "cross-attention" claim.

## Part 5. Checklist

- [ ] Q1: replay = archived/looped footage (p. 4); streaming claim narrowed; single-pass memory-building cost explained
- [ ] Q2: capacity, update equation, no τ, replacement (rebuild), per-block memory, skipped tokens keep the host's K/V
- [ ] Q3/Q4: not measured; indirect evidence from p. 5
- [ ] Q5: plot the host's sweep and PSM's γ sweep; other sweeps not available
- [ ] Q6: unfused matching, batch-1 overhead, host shows the same, matching FLOPs added to the count
- [ ] Q7: T ≤1.2 points; γ per task; Kinetics-400 issue disclosed
- [ ] Q8: staleness bounded by the rebuild period; similarity floor as future work
- [ ] Related work: LLM-side KV/token methods vs. vision-backbone recomputation
- [ ] ≤1,500 characters, no links, consistent with Reviewers 1 and 2
