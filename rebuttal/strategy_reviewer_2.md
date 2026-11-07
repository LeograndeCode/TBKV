# Rebuttal strategy — Reviewer 2 (score 4, confidence 3)

**Rules** (AAAI-27 email): one response per review, at most 1,500 characters, no links, deadline 31 Oct 2026 (AoE). The research behind this strategy (AAAI policy, Parikh et al., Gao et al.) is in `strategy_reviewer_1.md` §1 and applies here unchanged.

## Reviewer profile

- **Score:** 4, weak reject. The reviewer is constructive and engaged with the technical details.
- **Specific points for rebuttal:** "See weaknesses", i.e. W1–W5.
- **Decisive sentence:** "the current evidence does not sufficiently establish that persistent semantic retrieval provides a better accuracy–efficiency trade-off than simpler budget reduction."
- **Strengths they credit:**
  - the motivation;
  - the training-free interface;
  - the reporting of latency, memory and limitations.

  Thank them for these. They also show the reviewer values honesty.

## Part 3. Classification

| Weakness | Category | Real concern | How to resolve | Convincing evidence | Score impact |
|---|---|---|---|---|---|
| W1: long-term reuse not shown | Missing evidence; claim exceeds evidence | Paper claims long-gap reuse but only tests ~10 s clips | Narrow the claim + state the limitation (no new experiments allowed) | Honest reason for the benchmarks (comparability with host and baselines) | Medium |
| W2: no matched-budget host comparison | **Unfavourable result** (central) | Does PSM beat just lowering k? | Concession + narrowed claim; the layered framing explains intent but does not answer the question | p. 5 already reports ≈1.5 mAP recovered by budget-matched Eventful | **Highest.** It's the reviewer's stated reason to reject |
| W3: description inconsistent | Missing detail / factual error | Which method was evaluated? | Clarification (easy) | Three precise answers tied to Algorithm 1 | Medium; easy win |
| W4: latency gain small; large K400 drops | Unfavourable result; partly explained by an implementation issue | Is the efficiency real, and why does K400 collapse? | Explanation (latency) + disclosure (K400) | Table 1 (faster than dense, parity with host); class-token issue explains the flat ~58% | Medium |
| W5: readout explanation uncontrolled | Weak evidence for an interpretation | Is the readout claim causal? | Concession; restate it as a hypothesis | Task choice explains comparability, not causality | Low to medium |

## Part 4. Strategy

- **A. Approach.**
  - **Order:** start with the answers that are clear and favourable (W3, W4 latency), then concede W2 cleanly, then W1 and W5 briefly.
  - **Tags:** label each paragraph with its W-number so the reviewer can tick them off.
  - **Consistency:** Reviewer 1's response uses the same wording on the method, the budget-matched point and Kinetics-400. The area chair reads both, so they must match exactly.
- **B. Priority.** W2 matters most but has the weakest available answer. W3 and W4 are easy and reduce the reviewer's uncertainty. W1 and W5 are about scope, so answer them in one or two sentences each.
- **C. Evidence.**
  - **Essential:** Algorithm 1 l. 14; p. 5 budget-matched sentence; Table 1 latency; the Kinetics-400 issue.
  - **Recommended:** the reason for choosing these benchmarks (Eventful and the baselines evaluate on VID/K400).
  - **No new numbers.**
- **D. Central criticism (W2).** Of the six options, use:
  - narrow the claim;
  - acknowledge the gap;
  - revise the contribution statement ("training-free add-on; γ = 0 is the host").

  Avoid "we omitted the comparison because it's a layer": it reads as evasive, because the comparison is exactly how a layer's value is tested. Don't cite numbers from outside the paper.
- **E. Experiments.** None in the rebuttal. For the camera-ready, the decisive test is PSM on top of a lower-budget host versus that host at matched cost. Others: a long-video/revisit protocol; Kinetics-400 with the fix.
- **F. Style.**
  - Answer "yes / not quite" first, then the evidence.
  - Don't invent justifications. *Illustrative, avoid:* "Long-term reuse is implicitly demonstrated by the replay protocol." *Illustrative, prefer:* "Our benchmarks have short clips, so we show reuse within videos only; we will scope the claim accordingly."

## Part 5. Checklist

- [ ] W1: benchmark choice explained honestly; claim narrowed to within-video reuse; long gaps and revisits marked untested
- [ ] W2: layered intent stated; budget-matched gap conceded (p. 5); point to be plotted; claim narrowed; combined rule as future work
- [ ] W3: fraction, no τ; frozen + periodic rebuild; skipped tokens reuse the host's K/V (Algorithm 1 l. 14)
- [ ] W4: latency in context (faster than dense, parity with host, no speed-up claim); Kinetics-400 drops explained by the implementation issue
- [ ] W5: readout presented as a hypothesis; task choice explained by comparability
- [ ] No links; ≤ 1,500 characters; wording consistent with Reviewer 1
