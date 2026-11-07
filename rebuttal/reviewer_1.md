Thank you for your careful review and for appreciating our content-based design.

**1. Method:** No similarity threshold is used; τ on p. 3 is a leftover from an earlier version, sorry for the confusion. Algorithm 1's reuse fraction γ instead fixes how many tokens are skipped within budget k. The memory holds the tokens the host recomputes in the first W=4 frames, merged by T bipartite passes with update d ← (d + Σa)/(1 + n), where a are the n tokens matched to d. It is rebuilt every N frames (not used on our ~10 s clips). Later frames skip the γk proposed tokens most similar to the memory. We use γ=0.25, T=6 for detection (0.95, 2 for K400), chosen on a 10% subset (p. 4). The revision will add this.

**2. Numbers:** Thank you; we will make these numbers consistent throughout.

**3. Practicality:** For latency, PSM is 14% faster than dense (60.0 vs. 70.1 ms) and close to the host. For memory, it adds ~5 MB, independent of video length; the other 2,324 MB is the host's (p. 5). For accuracy, it costs 1.9 mAP vs. the host at 672 (3.9 at 1024) for Table 1's lowest compute (60.7 → 46.6 GFLOPs/frame). On the trade-off, we agree a budget-matched host recovers ~1.5 mAP (p. 5), and we will narrow our claims. PSM is a drop-in, training-free layer (γ=0 is the host) for recurring content, e.g. stationary cameras (pp. 4–5).

**4. Presentation:** Thank you; we will fix the Fig. 1 caption, if possible add the granularity table, and revise the headings.

Thank you again!
