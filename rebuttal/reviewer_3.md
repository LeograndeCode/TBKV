Thank you for a thorough, generous review; sorry the character limit forces brief answers.

**Q1:** Not yet: detection uses replay (archived footage, pp. 4–5), so we will narrow the streaming claim; a single pass only adds one merge per block.

**Q2:** Each of T merging passes halves the bootstrap tokens, d ← (d + Σa)/(1 + n) over the n tokens a matched to d, giving 52 per block at 672, rebuilt every N frames (unused in our runs). No threshold (τ is a leftover): we skip the γk proposed tokens most similar to the memory, which keep the host's K/V.

**Q3:** Short clips allowed no recurrence-gap study, but one early memory kept the full saving over a few hundred frames; with rebuilds, a long video is a chain of such segments. Dynamic vs. static scenes remain open.

**Q4:** Unmeasured, but a budget-matched host recovering ~1.5 mAP (p. 5) suggests some skips hurt detection.

**Q5:** If possible, we will plot Eventful's six-budget curve (already run) with PSM; other baselines have no sweeps.

**Q6:** We report latency to show it stays near the host and 14% below dense, not to claim a speed-up; unfused matching likely explains the gap (p. 5).

**Q7:** T changes mAP by ≤1.2 for γ ≤ 0.75 (p. 6); γ is per task, untested across subsets.

**Q8:** A stale memory lasts only until the next rebuild, at most N frames.

ProtoKV, VLMCache, StateKV and STTM target (video) LLM tokens and KV caches, not a vision backbone like PSM; we will discuss them. Thank you again!
