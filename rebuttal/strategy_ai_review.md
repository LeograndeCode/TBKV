# Rebuttal strategy — AI review (no score)

Prepared 27 Oct 2026. Labels: **[Official]** is published conference policy, **[Advice]** is experienced researchers' guidance, **[Judgment]** is my own assessment. The general research on rebuttal practice is in `strategy_reviewer_1.md` Part 1 and applies here unchanged.

## Part 1. What is specific to the AI review

| Finding | Source | Type |
|---|---|---|
| The AI review gives no score and no accept/reject recommendation. Decisions are made only by humans (PC, SPC, AC). | AAAI-26 AI-review pilot FAQ, Q2, Q7, Q9 | [Official, AAAI-26] |
| Authors may rebut the AI review like a human review. The AI review is not updated, but the human reviewers read the response and take it into account in the discussion. | Same FAQ, Q18 | [Official, AAAI-26] |
| "Major issues that warrant oversight from the conference staff can be flagged using the 'comment' field, which is visible only to SPCs and above." This is the confidential comment. | Same FAQ, Q18 | [Official, AAAI-26] |
| After the discussion, an AI summary of the reviews and the discussion points goes to the SPC. | Same FAQ, Q7 | [Official, AAAI-26] |
| The rebuttal should focus on substance, especially factually incorrect claims, not tone. No new results. | AAAI-26 rebuttal FAQ | [Official, AAAI-26] |
| AAAI-27 keeps the two-phase process; the PC and SPC discuss "taking author feedback into account". The response limits (1,500 characters per review, no links) come from the AAAI-27 email. | AAAI-27 review-process page; email | [Official] |
| SPCs/ACs were told to cross-check AI-raised issues against the human reviews, not to defer to them. | Secondary write-up of the AAAI-26 pilot (arXiv 2604.13940) | [Advice/secondary] |

**[Judgment] What this means.** The AI review can't vote, but it is the most detailed critique the SPC sees, and its points overlap with R2 (attribution), R3/R4 (long gaps, latency) and R1 (method text). The SPC will cross-check them, so every answer must match the human-review responses. Arguing against points the paper's own pages confirm would cost credibility for nothing.

## Part 2. Classification of the criticisms

| Point | Category | Real concern | Resolution | Impact |
|---|---|---|---|---|
| W1 Attribution ("three quarters" counts host savings; no random or matched-k control) | Fundamental weakness + presentation error | Is the gain due to the memory or to skipping more? | **Correct** the "three quarters" wording (PSM's own share is 60.7 → 46.6 GFLOPs/frame, 8.1 of the 73.3 points vs dense). **Acknowledge**: p. 5's matched-budget point favours the host, and no random-skip control exists. Claim composability, not superiority. | Highest; same as R2 W2 |
| W2 Long-gap recurrence untested | Missing evidence | Is "persistent" shown? | **Limitation**: within-clip reuse after a 4-frame bootstrap only; gap, age and drift untested | High; same as R3/R4 |
| W3 Complexity (mixed FLOP conventions; Eq. 4 assumes quadratic savings) | Factual error in the paper | Do the equations describe what runs? | **Correct**: skipped tokens keep the host's K/V (Alg. 1, l. 14), so savings are linear in γk. Table 1 is measured by counters, not derived from Eqs. 3–4. Disclose that the counters omit the matching (~0.25 GFLOPs/frame), contrary to p. 4. | Medium; an easy credibility win |
| W4 Novelty (MC-ViT, VidToMe, LITE) | Positioning | Is it a "new primitive"? | **Accept** the narrower framing; cite all three | Medium-low |
| W5 Latency (no host-relative speed-up; behind pooling at 1024) | Unfavourable result | Practical value? | **Acknowledge**; the paper already says this (pp. 5–6). Same wording as R2/R3/R4 | Medium |
| W6 Recognition far below dense | Unfavourable result (plus the undisclosed K400 issue) | Is the K400 point practical? | Report the dense gap; narrow the claims. Don't defend the K400 numbers (§11). | Medium |
| Minor 1–6 | Presentation errors | — | Accept all | Low, but cheap |
| Suggestions (failure taxonomy, adaptive γ, readout schematic) | Future work | — | Adaptive γ is future work (author notes §4). Don't promote the readout hypothesis: it rests on the K400 results. | Low |

**Verified against the paper and code:** "uncounted" bootstrap (p. 4: "unscored, uncounted"); interpolated matched point (p. 5); "three quarters" (p. 6); Eq. 4 and the break-even M < 2γC (p. 3); Table 1 (pooling at 672: 52.8 GFLOPs, 49.0 ms, 79.48); the paper never mentions a random control; the code matches on the LayerNorm-ed block input (`src/psm/blocks.py`), so the minor "query" point is correct.

## Part 3. Strategy

- **A. Approach.** Open with thanks and the strengths the review itself lists (cost accounting, honest failure regimes). Then answer each weakness in one or two sentences, starting "Agreed" where the paper confirms the point, followed by the concrete fix. End with one line accepting all minor points.
- **B. Priority.** W1 > W3 > W2 > W5 > W6 > W4 > minor. W3 is the only point where the answer improves the paper's standing: the reported costs are measured, not derived.
- **C. Evidence.** Paper pages only. No new numbers: the k=384 run (81.40) stays out, and we cite p. 5 instead.
- **D. Central criticism (attribution).** Of the six options: narrow the claims, acknowledge the gap, and revise the contribution statement (composable, training-free layer). Don't argue from FLOPs alone, and don't invent deployment advantages.
  - Not the 672 pooling comparison: pooling is also 11 ms faster.
  - Not the T ablation: a flat T curve supports the attribution concern rather than answering it.
- **E. Experiments.** None promised. A random-skip control at the same γ is the decisive test; mention it only as missing.
- **F. Style.** Same wording as the human-review responses on the matched budget, long gaps and latency.
  - *Illustrative, avoid:* "The memory clearly drives the savings."
  - *Illustrative, prefer:* "Agreed: the paper does not isolate the memory from extra skipping."

## Part 4. Action plan

- **Immediate corrections (revision):** the "three quarters" wording; Eqs. 3–4 and the break-even for the implemented path; adding the matching to the counted FLOPs; "new primitive" → reuse policy; MC-ViT, VidToMe and LITE cited; minor 1–6.
- **Implementation checks:** done (bootstrap accounting, query tensor, counters, absence of a random control). The class-token check on the trained K400 checkpoint is still open: the weights are not on this machine.
- **Experiments:** none in the rebuttal. Camera-ready candidates: a random-skip control, PSM on a lower-budget host, K400 with the fix.
- **Fundamental concerns:** attribution and long gaps are scope limits that the revision states; they can't be resolved in the rebuttal.

## Checklist

- [ ] Strengths acknowledged; "Agreed" where the paper confirms the point
- [ ] Attribution: own share stated, "three quarters" dropped, p. 5 matched point and missing random control acknowledged
- [ ] Complexity: linear savings for the implemented path; counters, not equations; matching added
- [ ] Recurrence, latency, recognition: same wording as R1–R4
- [ ] No K400 cause; no new numbers; no links; ≤1,500 characters

Sources: AAAI-26 FAQ for the AI-Assisted Peer Review Process Pilot Program (aaai.org, Aug 2025); AAAI-26 FAQ for authors, rebuttal (aaai.org); AAAI-27 review process (aaai.org); Biswas et al., "AI-Assisted Peer Review at Scale: The AAAI-26 AI Review Pilot", arXiv 2604.13940.
