# Rebuttal for Submission 38867

One file per reviewer. Each file thanks that reviewer and can be posted on its own.

| File | Reviewer | Their score | Main concern |
|---|---|---|---|
| [reviewer_1.md](reviewer_1.md) | Reviewer 1 | 4 (reject) | Method description, inconsistent numbers, cost |
| [reviewer_2.md](reviewer_2.md) | Reviewer 2 | 4 (reject) | Is memory better than just recomputing fewer tokens? |
| [reviewer_3.md](reviewer_3.md) | Reviewer 3 | 6 (weak accept) | Real streaming tests, full curves, latency |
| [reviewer_4.md](reviewer_4.md) | Reviewer 4 | 5 (weak reject) | Novelty against related methods, trade-offs |
| [ai_review.md](ai_review.md) | AI review | none (AI reviews are unscored) | Is the saving due to the memory or to skipping more? |
| [confidential_comment.md](confidential_comment.md) | SPC and AC only | — | Consistency across reviews |

## What changed in the paper, in short

1. **Method text.** There is no threshold: the fraction γ most similar to the memory is skipped. The memory is built from the first 4 frames and refreshed after a fixed number of frames. The refresh is implemented but never triggers on the ~10-second benchmark videos. The reuse path is described as the code runs it.
2. **Kinetics-400 results.** The responses acknowledge the larger losses at higher budgets and narrow those claims. The class-token bug is in no response (author's decision); `author_notes.md` §11 records it. It is not in the confidential comment either (author's decision, 27 Oct 2026).
3. **Detection results.** At the same cost, a lower-budget Eventful beats PSM. The paper will say so and present combining memory similarity with the host's rule as the next step.
4. **Smaller fixes.** The Figure 1 caption, the Figure 2 reference, the missing T table, the 49% / 2.5 numbers, and the cost accounting.

## Before sending

- Re-run the class-token check on the trained checkpoint (CPU, a few minutes) before the camera-ready. No response depends on it. The earlier check used random weights.
- Responses promise only fixes in the revision or appendix material built from existing runs. No new experiments.
- Paste each response into its box and check the counter: at most 1,500 characters and no links (AAAI-27 rules).
- Verify the related-work citations (ProtoKV, VLMCache, StateKV, STTM).
