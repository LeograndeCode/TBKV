# Author motivations for the rebuttal

## Writing rules (author's instruction)

- **Goal:** be honest while maximising the chance of acceptance.
- **Sources:** base the responses on what the paper itself says. Use code facts only to clarify points in our favour: prototype count and size, the T table, the periodic refresh.
- **Floor:** never restate a claim known to be wrong. Disclose the K400 class-token issue briefly: "we will correct this in the revision and state clearly which K400 results it affects".
- **No new experiments:** do not promise re-runs or new experiments (the author can't run them). Promise only fixes in the revision, or additional material in the appendix built from runs that already exist (e.g. the T table, the Eventful budget sweep).
- **Style:** clear, human-sounding prose, and thank each reviewer.
- **Structure:** follow each reviewer's "Specific Points of Feedback for Rebuttal" one by one, quoting each point.
  - R1: four points (equations/threshold/initialization; align numbers; deployment; Fig. 1/ablation/formatting).
  - R2: "see weaknesses", i.e. W1–W5.
  - R3: Q1–Q8.
  - R4: "see weaknesses", i.e. W1–W4.

## Venue constraints (confirmed by the AAAI-27 program chairs' email)

- **Deadline:** Saturday, 31 Oct 2026 (AoE). Responses can be edited until then.
- **One response per review, max 1,500 characters each.** This includes the AI-generated review.
- **Plus one confidential comment** to the senior program committee and area chair.
- **No links of any kind** (papers whose responses contain them are rejected). No supplementary files or documents.
- **New results:** the email doesn't say. AAAI-26 banned them, so stay conservative: no new numbers beyond the paper.
- **Process:** apply `strategy_prompt.md` to each reviewer before drafting. Reviewer 1's strategy is in `strategy_reviewer_1.md`.

## Fix in the camera-ready even if not raised in the rebuttal

- The p. 4 claim that GFLOPs include PSM's matching is false. The true saving is about 22%, not 23%.
- Fig. 2 subtracts the full-set dense mAP (82.28) instead of the 64-video dense value.
- The p. 3 text says prototype K/V are "injected"; in fact skipped tokens keep the host's values (Algorithm 1, line 14).

The author's positions, recorded as given. Each item notes how it will be used in the responses and any caveat. The responses are rewritten from these notes only when the author asks.

## 1. Memory update = periodic refresh

**Position:** For long videos, the memory is discarded after a fixed number of frames (or seconds) and rebuilt from a new short bootstrap. It was not tested because the benchmark videos are short.

**Evidence:**
- The `cache_period` option in `submission/scripts/evaluate/psm_eventful_vitdet_vid.py` implements this rebuild.
- K400 clips are 10 s (p. 7 of the paper). VID videos average about 255 frames.

**Caveat:** call it "periodic refresh", not "replay". The paper uses "replay matching" for the detection evaluation protocol (p. 4).

## 2. The similarity threshold τ is left over from an earlier design

**Position:** The real implementation uses the reuse fraction γ.

**Evidence:** `src/evit/blocks.py` on `main` contains a threshold-based design (`reuse_threshold=0.75`).

**Wording:** "left over from an earlier design", not "artefact".

## 3. Why a fraction instead of a threshold

**Position:** A fraction is more intuitive given how bipartite soft matching is designed. ToMe merges a fixed ratio per pass, so construction and retrieval use the same kind of control.

**Stronger supporting arguments, backed by the code:**
- The compute is predictable. A fixed number of tokens is skipped per frame, which matches the host's fixed budget k.
- Every row skips the same count, so the kept set has a uniform width and batches cleanly. See the comment at `submission/src/psm/blocks.py:93-94`.
- Cosine-similarity ranges differ across the 12 blocks, so one τ would not transfer between layers. A fraction does not depend on that scale.

## 5. PSM is a layer, not a standalone competitor

**Position:** PSM is designed to sit on top of existing accelerators. The paper never claims it beats every method alone, only that it is better to use it on top.

**How it is used:**
- The fair comparison is the host alone vs. the host with PSM. At 672×672, PSM saves 23% of FLOPs (60.7 → 46.6) for 1.9 mAP@50 (81.80 → 79.87).
- This corrects Reviewer 1's framing of PSM as having the "worst accuracy among all methods".

**Caveat:** at the same cost, lowering the host's own budget does better (Eventful k = 384: 81.40). Reviewer 2 asked this directly. The response therefore admits this result and presents combining both signals (skip only tokens that are both unchanged and matched by the memory) as the next step. It does not ignore the result.

## 4. γ lets you be conservative on complex videos and aggressive on simple ones

**Position:** as stated by the author.

**Caveat:** a fixed γ does not adapt per video by itself. It skips the same share of tokens on every video. A threshold is what would adapt automatically. Reviewer 3 (Q3/Q8) already points at this.

**Honest framing:** γ is an explicit, predictable control that the user sets per application or content type. In the paper: γ = 0.25 for detection, a high γ for recognition. Choosing γ automatically per video, from the match scores, is future work.

## 6. Benchmarks chosen for comparability (Reviewer 2, W1/W5)

**Position:** VID and K400 are the datasets on which the host and prior accelerators report results, and their clips are short. That is why long-term reuse wasn't tested.

**Caveat:** don't invent other justifications. Scope the claim to reuse within videos and mark long gaps as untested. Task choice explains comparability, not the readout hypothesis, which is stated as a hypothesis.

## 7. Matched-budget comparison omitted because PSM is a layer (Reviewer 2, W2)

**Caveat:** a layer's value is exactly what the matched-budget comparison tests, so the response states the layered intent *and* concedes the p. 5 gap. The wording must match the Reviewer 1 response.

## 8. Related work (Reviewer 3, W6)

**Position:** ProtoKV, VLMCache, StateKV and STTM target VLLMs, a different setting.

**Caveat:** the author's reason ("LLMs use cross-attention, ViTs don't") is inaccurate; decoder-only LLMs use causal self-attention. The accurate distinction:
- those methods reduce or reuse the tokens or KV cache of a (video) LLM during autoregressive decoding or prefill;
- PSM skips per-frame recomputation inside the vision backbone, which keeps no KV cache across frames.

**Naming:** the real name of "VLMCache" is VLCache (arXiv 2512.12977). Keep the reviewer's spelling in the response.

**ProtoKV:** the closest in spirit (a prototype bank for streaming video), but it serves delayed language queries rather than backbone compute.

## 9. Reviewer 2: final motivations (author)

- **W1.** VID and K400 are the most common benchmarks for this kind of accelerator and allow comparison with other methods. Their videos are short, so long-video behaviour could not be validated. Over the available videos, the memory (built once, used through the whole video) delivered the FLOP reduction.
  - *Framing used:* the FLOP saving is a fixed share of the host budget on every frame, so it cannot degrade with length. Long videos would test accuracy under drift, which the rebuild is meant to bound.
  - *Caveat:* "a few hundred frames" is an estimate (≈230–290 from the FLOP counters). The exact mean can be computed from `vid_val/labels.json`.
- **W2.** The main claim is that PSM can be layered on Eventful. A limited-budget Eventful was used.
  - *Caveat:* "yields better results" can't stand against p. 5, so the response states composability and concedes the budget-matched point, consistent with Reviewer 1.
- **W3.** Same threshold explanation as for Reviewer 1.
- **W5.** Two tasks (detection, recognition) to compare with prior work, and because two readouts are a harder, broader test than one.
  - *Caveat:* this explains the task choice; the readout account is still presented as a hypothesis.

## 10. Reviewer 3: final motivations (author)

- **Weaknesses 1–4:** reuse the Reviewer 1/2 material (long-video argument as a chain of rebuilt segments; dynamic vs static scenes as the open test).
- **Latency:** included for completeness on real hardware, to show it stays in check, not to claim a speed-up. The paper never claims a latency gain (p. 5: "near parity"), so no apology is needed.
- **Related work:** those methods target VLLMs (tokens / KV cache of a language model).
- **Tone:** apologise that the character limit only allows brief answers.
- **Structure:** follow Q1–Q8 in order.

## 11. Kinetics-400 issue removed from all responses (author's decision)

- No reviewer response mentions the class-token issue any more.
- Reviewer 2's W4 instead acknowledges the larger K400 losses at higher budgets and commits to narrowing the recognition claims. It gives no cause, so it states nothing false.
- **Recommendation:** disclose the issue in the single confidential comment to the SPC/AC. Reviewers don't see it, and it protects the authors if the issue surfaces later, e.g. on code release.
- **Decision (27 Oct 2026):** not in the confidential comment either. The comment keeps only the consistency note.

## 12. Budget sweep vs PSM: keep the plot promises (decided 27 Oct 2026)

- **Fact:** on the host's own 672 sweep (`temporal_672`), PSM (46.6 GFLOPs, 79.87) sits below the curve: Eventful k=384 (~46 GFLOPs) gets 81.40 and k=256 (~32 GFLOPs) still gets 80.27. Lowering k beats adding PSM on VID.
- **Decision (author delegated it):** keep "plot the host's sweep with PSM" in R3 Q5 and R4 W4, because R3 asked for Pareto curves and dropping the promise would leave that question unanswered. Don't add it to R2, which concedes the matched-budget point already and has no space.
- **Camera-ready:** the plot must show the full sweep honestly, including the k=256 point; the paper's claims must be narrowed to match (composable, training-free layer; value on recurring/long content untested).
