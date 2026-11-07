Thank you for a precise review and for valuing our cost accounting and failure-regime reporting; we agree with most points.

**Attribution:** Agreed. Our only matched-budget control, p. 5's interpolated host point, favors the host by ~1.5 mAP, and no random-skip control was run. We will state PSM's own share (60.7 → 46.6 GFLOPs/frame, 8 of the 73 points vs. dense), drop the "three quarters" reading and claim composability, not superiority.

**Recurrence:** Agreed: both protocols test reuse within short clips after a bootstrap, not after long gaps. We will scope the persistence claim; gap, cache age and drift are untested.

**Complexity:** Agreed. Skipped tokens keep the host's K/V (Alg. 1, l. 14), so savings are linear in γk; we will rederive Eqs. 3–4 and the break-even for this path with one FLOP convention. Table 1 uses op counters, not these equations; the counters omit matching (~0.25 GFLOPs/frame), which we will add.

**Novelty:** We will cite MC-ViT, VidToMe and LITE and present PSM as a training-free, content-addressed reuse policy, not a new primitive.

**Latency:** We claim no speed-up: PSM is near host latency and behind the other gated methods, and at 1024 spatial pooling is faster and more accurate (pp. 5–6).

**Recognition:** We will report the dense gap beside every K400 result and narrow those claims.

**Minor:** All accepted: the 49% figure, Fig. 1 caption, query wording, frozen memory, subsample overlap, and no reading within noise.

Thank you again!
