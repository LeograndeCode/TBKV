Thank you for a thoughtful review and for valuing our training-free interface and honest reporting.

**W1:** VID and K400 are the standard benchmarks, but their short clips prevent a long-video test, as we will state. Over videos of a few hundred frames, one early memory kept the full saving, a fixed share of the host's budget per frame. With rebuilds every N frames, a long video is a chain of such segments, so we expect results to hold; dynamic vs. static scenes remain open.

**W2:** Our main claim: PSM layers onto Eventful without retraining (γ=0 is the host), cutting 60.7 to 46.6 GFLOPs/frame. We agree the key test is the host at equal cost: a budget-matched Eventful recovers ~1.5 mAP (p. 5), so we will narrow our claims; combining memory and change signals is next.

**W3:** Sorry for the inconsistency: no threshold is used; τ (p. 3) is a leftover from an earlier version. Algorithm 1's fraction γ skips the γk proposed tokens most similar to a memory rebuilt every N frames (unused in our runs); skipped tokens keep the host's K/V (l. 14). We will rewrite the method text.

**W4:** We report latency to show it stays near the host and 14% below dense, not to claim a speed-up; even the host turns 65% fewer FLOPs into only 16% lower latency. We acknowledge the larger K400 losses at higher budgets and will narrow those claims.

**W5:** Both tasks follow prior work and test two readouts, not one; the readout explanation stays a hypothesis, as we will say.

Thank you again!
