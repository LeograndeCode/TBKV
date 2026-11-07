Thank you for your clear summary and for recognizing our formulation, implementation and broad evaluation.

**W1:** These methods differ in purpose. Deep Feature Flow warps key-frame features with a trained optical-flow network, tying reuse to position; MeMViT and TTM are trained with a memory that adds context to improve predictions. PSM is training-free and uses memory only to decide, by content, which host recomputations to skip, as another reviewer notes.

**W2:** Detection uses replay, modeling archived footage (pp. 4–5), so we will narrow our streaming claims. Our clips are short, but one early memory kept the full saving over a few hundred frames; with rebuilds every N frames (not used in our runs), a long video becomes a sequence of such segments. Dynamic vs. static scenes remain open.

**W3:** Each block keeps one bounded memory: the bootstrap tokens merged by T averaging passes, 52 per block at 672, rebuilt every N frames rather than evicted. No threshold is used (τ is a leftover); PSM skips the γk proposed tokens most similar to the memory, as the method text will clarify.

**W4:** We report latency to show it stays close to the host and 14% below dense, not to claim a speed-up; optimized matching kernels are future work. Accuracy drops more at 1024 and at larger γ, so γ should stay low, and a budget-matched host recovers ~1.5 mAP (p. 5). We will narrow our claims and, if possible, add the budget sweeps.

Thank you again!
