# Rebuttal strategy — Reviewer 1 (score 4, confidence 3)

Prepared 26 Oct 2026. Sources are listed at the end. Labels: **[Official]** is published conference policy, **[Advice]** is experienced researchers' guidance, **[Evidence]** is empirical research, and **[Judgment]** is my own assessment.

---

## 0. UPDATE: rules confirmed by the AAAI-27 email (they replace the assumptions below)

- **Deadline:** 31 Oct 2026, anywhere on earth.
- **Length:** one response per review, at most **1,500 characters** each.
- **Confidential comment:** one, to the senior program committee and area chair.
- **No links, no files.**
- **New results:** the email says nothing. Stay conservative.

The rest of this strategy still applies, but the plan becomes per-review responses with no shared 2,500-character budget.

## 0 (original). Two constraints that changed the whole plan

**1. Deadline. Check OpenReview today.**
- [Official] The AAAI-27 site lists the author-feedback window twice, with two different ranges: "October 19–25, 2026" and "October 24–31, 2026". Deadlines are anywhere-on-earth (UTC−12).
- Today is 26 October. If the first range applies, the window has closed or is closing now.
- The submission page on OpenReview shows the real deadline. Check it before anything else.

**2. Length and content rules.**
- [Official, AAAI-26; AAAI-27 not yet published] AAAI-26 allowed **a single rebuttal of at most 2,500 characters covering all reviews**. Its other rules:
  - **no new results**, "even if the reviewers asked for them";
  - **no URLs**;
  - markdown is allowed;
  - authors may respond to the AI review;
  - the response should focus on "the substance of the reviews … particularly … any factually incorrect claims".
- AAAI-27's pages only say the feedback is "an opportunity to address questions or factual mistakes in reviews". The OpenReview response box shows the actual limit.

If the AAAI-26 rules carry over, our current per-reviewer drafts (about 3,000–7,000 characters each) cannot be submitted as written. The whole response, covering all four reviewers, must fit in roughly 2,500 characters, which is about 400 words. Everything below assumes that limit. If the box allows more, the same priorities apply with more room.

---

## Part 1. What the research says

| Question | Finding | Type |
|---|---|---|
| Purpose of a rebuttal | AAAI: address questions and factual mistakes; no new results. CVPR template: correct factual errors and supply requested information, not add new contributions. | [Official] |
| How it is read | AAAI: the program committee (PC) and senior PC discuss with the feedback "into account"; scores are not averaged, and senior members decide "after considering all available information". NeurIPS asks reviewers to read responses and explain any score change. | [Official] |
| Who to write for | Write so that a neutral reader, effectively the area chair (AC), who may read only the reviews and the rebuttal, can tell whether each concern was addressed. | [Advice: Parikh, Batra & Lee] |
| Structure | Start with what reviewers praised. Quote each concern, then answer directly ("Yes / No / Not quite"). Lead with the biggest concerns you have good answers for. Merge shared concerns. | [Advice: Parikh et al.] |
| Experiments you can't run | Say so plainly. AAAI explicitly allows "clarifying why some experiments are not needed". | [Official AAAI-26] [Advice] |
| Tone | Conversational, receptive. Don't critique the review's tone or fairness. | [Official AAAI-26] [Advice] |
| Does it change outcomes? | Effects on final scores are statistically significant but small, mostly for borderline papers. Final scores depend largely on the reviewer's own initial score and on the other reviewers' scores (conformity). | [Evidence: Gao et al., NAACL 2019, ACL-2018 data] |

**[Judgment] What this means for this paper.**
- The scores are 4 / 4 / 6 / 5, so the paper is below borderline.
- The evidence does not support expecting a rebuttal to flip Reviewer 1. Realistic goals are:
  1. fix the factual errors in the review, so the AC discounts them;
  2. show that reproducibility is now handled;
  3. keep credibility, so that Reviewer 3 (the only positive score) can champion the paper in discussion.
- Losing credibility, for example by overclaiming something the AC can check against the paper, is the biggest avoidable risk.

---

## Part 3. Classification of Reviewer 1's criticisms

### W1. Incomplete method description: equations, threshold, initialization

- **Categories:** missing method detail; reproducibility; also a factual/presentation error, because the τ text does not match the method used.
- **Real concern:** can anyone reproduce PSM? The paper describes τ, online updates and injected K/V, but Algorithm 1 uses γ, so the reader can't tell which method was evaluated.
- **How to resolve:** clarification and correction, in text only. No experiment is needed.
- **Convincing evidence:**
  - one compact equation each for initialization, merge update and retrieval;
  - the parameter values (W = 4, T, ratio 0.5, γ, k);
  - an explicit statement that no τ is used, with the reason.
- **Impact on score:** moderate. It's easy to fix, and fixing it removes a reason to reject without creating a reason to accept.
- **Without unsupported claims:**
  - State that τ is left over from an earlier version. The repo supports this: `src/evit/blocks.py` on `main` has `reuse_threshold`.
  - Describe the refresh as part of the design that never triggers on ~10-second videos.
  - Don't claim the refresh was tested.
  - Don't claim that γ adapts per video. It doesn't; it's set per task.

### W2. Numerical contradictions (39% / 49%, 2.5 / 2.6)

- **Categories:** factual/presentation error. Hidden behind it is an **unfavourable result**: the Kinetics-400 numbers are affected by the class-token implementation issue.
- **Real concern:** sloppiness lowers trust in every number.
- **How to resolve:** correction, plus a one-sentence disclosure about Kinetics-400.
- **Convincing evidence:** state the correct values and where they come from (Table 3).
- **Impact on score:** low on its own. **High if mishandled.** Re-asserting the 39% / 2.6 figures as valid, when we know they are affected, is the single most damaging thing the rebuttal could do if it later surfaces.
- **Without unsupported claims:**
  - "49% is a typo; the abstract and introduction will match the tables."
  - "We found an implementation issue affecting the Kinetics-400 experiments (ViViT's class token was not recomputed after the first frames when γ > 0); the revision corrects it and marks the affected results."
  - Don't promise re-runs.

### W3. Unconvincing performance and hardware trade-offs

This is the central objection. The reviewer bundles five sub-claims, and each needs a different response.

| Sub-claim | Category | Verdict from the paper's own data |
|---|---|---|
| "Worst accuracy among compared approaches" | Possible misunderstanding (half) | At 672: false. Eventful + spatial pool scores 79.48 vs PSM 79.87 (Table 1). At 1024: true (75.57 is the lowest). |
| "Huge memory footprint" | **Misunderstanding** | The paper (p. 5) says the host's buffers account for 2,324 MB and the prototype cache adds ~5 MB. The footprint belongs to the host. |
| "Worse latency" | Misunderstanding (half) | PSM is faster than dense (60.0 vs 70.1 ms) and roughly at parity with its host (58.9 ms). It is slower than the other gated baselines (49.0–55.1 ms). |
| "Unfavourable accuracy–compute trade-off" | **Unfavourable empirical result** (genuine) | The paper itself says a budget-matched Eventful "would recover roughly 1.5 mAP" (p. 5). |
| "Hard to justify for deployment" | Weak motivation / unclear practical contribution | Correct as things stand: there's no wall-clock gain over the host and no measured deployment advantage. |

- **Real concern:** "Why would anyone use this?"
- **How to resolve:** clarification for the two misread metrics (memory, latency); narrowing the claims and a limitation statement for the trade-off. It can't be resolved in the rebuttal with new results, because those aren't allowed.
- **Convincing evidence:**
  - Table 1 numbers that are already in the paper;
  - the "layer, not competitor" framing, supported by the paper's own definition (Algorithm 1; γ = 0 reduces to the host);
  - an honest statement of what has not been shown.
- **Impact on score:** **highest.** This is what moved the score to 4.
- **Without unsupported claims:**
  - Don't claim a deployment scenario where PSM wins; none has been measured.
  - Don't argue that FLOPs alone matter; Reviewer 1 has already rejected that.

### W4. Presentation: Figure 1 caption, missing granularity table, formatting

- **Categories:** factual/presentation error; missing evidence (the granularity table).
- **How to resolve:** correction.
- **Convincing evidence:** "Fixed: upper-left is better." For the table, point to the paper's existing statement (p. 6: "Sweeping T ∈ {2, 4, 6, 8} at fixed γ moves mAP@50 by at most 1.2 points") and commit to adding it to the appendix.
- **Caution on the table:** I'd *not* paste the 4×4 numbers into the rebuttal. AAAI-26 bans new results "even if the reviewers asked for them", and the table costs space. The summary statement is already in the paper.
- **Impact on score:** low, but cheap to fix, and it signals care.
- **[Judgment] A counter-risk to anticipate:** a flat T curve can be read as "the content of the memory barely matters; only γ, the amount skipped, matters". That supports the AI review's attribution concern. Don't claim more than "T is a safe default".

---

## Part 4. Strategy

### A. Overall approach

- **Structure: one combined response, organised by theme, not by reviewer.**
  - Under a 2,500-character total limit, separate per-reviewer essays are impossible.
  - [Advice] Merging shared concerns saves space. Tag each item with the reviewers it answers (e.g. "R1-W1, R2-W3, R3-Q2, R4-W3").
  - Reviewer 1's points map onto three shared themes: the method description (shared with R2, R3, R4), performance and the trade-off (shared with R2, R3, R4), and corrections (R1 only).
  - If OpenReview allows per-review responses with more space, use per-reviewer replies with the same priorities.
- **Acknowledge or rebut first?**
  - For W1, W2 and W4: acknowledge in one short clause, then state the fix.
  - For W3: correct the two factual misreadings first (memory, latency), then concede the trade-off point. Leading with the correction anchors credibility on facts the AC can verify. Conceding the real weakness right after shows you aren't spinning.
- **Balancing improvement against defence:** defend the *idea and the design* (training-free, a layer, γ = 0 is the host, a few MB of memory). Concede the *current empirical gap* rather than defend the trade-off.

### B. Prioritisation (for Reviewer 1's points)

| Rank | Issue | Importance to decision | Urgency | Ease | Can it move the assessment? |
|---|---|---|---|---|---|
| 1 | W3, factual parts (memory, latency, "worst") | High | High | Easy (paper data) | **Yes.** Removes misread negatives. |
| 2 | W1, method / threshold / initialization | High (reproducibility) | High | Easy | Somewhat. Removes a reject reason. |
| 3 | W2, numbers plus Kinetics-400 disclosure | Medium (trust) | High (must not be wrong) | Easy | Little upside; large downside if mishandled. |
| 4 | W3, trade-off and deployment | Highest | High | **Hard** (can't be fixed without new results) | Little. Aim to neutralise by narrowing claims. |
| 5 | W4, Figure 1, table, formatting | Low | Low | Trivial | Minimal. Keep to one line. |

### C. Evidence to prepare before writing

| Item | Status | Where it comes from |
|---|---|---|
| Update equations and pseudocode (init, merge, retrieval) | **Essential** | Code in `submission/src/psm`; already drafted |
| Threshold explanation (no τ; why γ) | **Essential** | Code plus your rationale; `src/evit` history |
| Initialization details (W = 4, the tokens the host selected, T passes) | **Essential** | Code |
| Reconciled numbers (39 / 2.6 from Table 3; delete 49 / 2.5) | **Essential** | Paper |
| Kinetics-400 class-token disclosure sentence | **Essential** | Code audit. Optionally confirm on the trained checkpoint first; the earlier check used random weights. |
| Runtime and peak-memory numbers | **Essential**, but cite only; no new measurements | Table 1 |
| Accuracy–compute comparison incl. budget-matched Eventful | **Strongly recommended** | Paper p. 5 (≈1.5 mAP) |
| Granularity ablation | Recommended: cite the p. 6 statement; table goes in the appendix | Paper plus stored runs |
| Evidence of a deployment advantage | **Not available.** Don't claim one. | — |

### D. The central performance criticism: evaluating the six options

| Strategy | Scientifically defensible? | Needs new evidence? | Likely to persuade Reviewer 1? |
|---|---|---|---|
| 1. Defend via FLOP reduction | Only partly. True, but the reviewer already discounted FLOPs without latency, and at equal cost the host's own budget matches it. | No | **Unlikely** if used alone |
| 2. Name a deployment scenario where PSM wins | Not at present. No measured scenario exists. | **Yes:** wall-clock gain with a fused kernel, energy per frame, long-stream results with refresh. | Unlikely, and it risks an unsupported claim |
| 3. Extra latency / memory / hardware measurements | Yes, but new results are not allowed in the AAAI rebuttal. | Yes (camera-ready only) | Not available now |
| 4. Narrow the paper's claims | **Yes** | No | **Likely** to reduce the penalty |
| 5. Acknowledge that the experiments do not show practical superiority | **Yes** | No | Builds credibility with the AC |
| 6. Revise the motivation: "training-free add-on and analysis of when content reuse helps" | **Yes** | No | Moderate; this is where the layered-design argument lives |

**[Judgment] Recommended combination: 4 + 5 + 6, framed by fixing the factual misreadings.**
1. Correct "huge memory" (the host's buffers; PSM adds ~5 MB) and "worse latency" (faster than dense, at parity with the host).
2. State the purpose: PSM is a training-free layer; γ = 0 *is* the host.
3. Concede: "at equal cost, a budget-matched host is ≈1.5 mAP better (p. 5); we will plot it and narrow the claims accordingly". Present the combined rule as future work.
4. Don't claim a deployment advantage.

**What would establish a deployment advantage (for a future paper, not the rebuttal):**
- PSM on top of a *lower-budget* host beating that host at matched cost;
- wall-clock speed-up with fused matching;
- long-video streaming results with the refresh;
- energy measurements.

### E. Experiments under time limits

- **For the rebuttal itself: none.**
  - [Official AAAI-26] New results are not allowed.
  - You've also said you can't run them.
  - [Official AAAI-26] Use the allowed move instead: "clarifying why some experiments are not needed" or why they are future work.
- **Private check before writing (not a result for the rebuttal):** run `check_frozen_readout.py` with the trained K400 checkpoint, about 10 minutes on CPU or GPU. It confirms that the disclosure sentence is accurate. If it *doesn't* reproduce, remove the disclosure and tell me.
- **For the camera-ready or a resubmission, in order of value:**
  1. PSM on top of lower-budget Eventful versus Eventful at matched cost. This is the decisive test of the layered claim.
  2. Count the matching FLOPs.
  3. Kinetics-400 with the class-token fix.
  4. Fused matching kernel latency.
  5. Scene-cut / refresh test.
- **If the granularity results look bad:** they don't. The curve is flat, but see the W4 counter-risk.
- **If new measurements reinforce the criticism:** report them and narrow the claims. Don't hide them.

### F. Writing style (examples below are illustrative, not rebuttal text)

- **Acknowledge without undermining.** Concede the specific point, not the whole contribution.
  - ✅ *Illustrative:* "Agreed: at equal cost, a lower-budget host is ≈1.5 mAP better (p. 5). The revision plots this point and scopes our claim to what PSM adds on top of a given host."
  - ❌ *Illustrative:* "We agree the results are not convincing and the method may not be useful."
- **Correct facts without sounding defensive.** Cite the paper, not the reviewer's error.
  - ✅ *Illustrative:* "Not quite: Table 1 attributes 2,324 MB to the host's buffers; the prototype cache adds ≈5 MB (p. 5)."
  - ❌ *Illustrative:* "The reviewer clearly misread Table 1."
- **Disagree respectfully.** "Not quite" plus the evidence, and stop there.
- **State what changed:** "We will add / will fix / will state" for revision items. Never "we have re-run".
- **Don't over-promise.** Only promise text fixes and appendix material built from existing runs.
- **Don't write a new paper.** Mention the combined rule in one clause, as future work.

---

## Part 5. Action plan for Reviewer 1

1. **First actions (today):**
   - Open OpenReview and confirm the deadline and the character limit.
   - Decide: single combined response or per-review responses.
   - Optionally run the class-token check on the trained checkpoint.
2. **Evidence to gather:** none new. Collect the paper citations: Table 1 rows; p. 5 memory sentence; p. 5 budget-matched sentence; p. 6 granularity sentence; Table 3 values.
3. **Manuscript changes to promise:**
   - rewrite the method section (equations, no τ, γ rationale, init, refresh);
   - align the numbers;
   - mark the Kinetics-400 results affected by the fix;
   - plot the budget-matched host point and narrow the claims;
   - Figure 1 caption;
   - granularity table in the appendix;
   - formatting.
4. **Order in the rebuttal:**
   1. factual corrections in W3 (memory, latency);
   2. method / threshold (W1);
   3. trade-off concession plus framing (W3);
   4. numbers and Kinetics-400 disclosure (W2);
   5. presentation fixes (W4), in one line.
5. **Structure:**
   - **Immediate corrections:** Figure 1 caption, 49% / 2.5, formatting, memory and latency misreads.
   - **Clarifications that needed the implementation (done):** no τ, γ rule, init, refresh, equations, prototype count.
   - **Experiments:** none in the rebuttal; camera-ready list in E.
   - **Fundamental concerns:** trade-off and deployment, handled by narrowing claims and a limitation statement.
6. **Checklist for Reviewer 1:**
   - [ ] Update equations stated (init, merge, retrieval)
   - [ ] "No threshold" stated, with the reason and the values of γ, T, W, k
   - [ ] Initialization protocol stated
   - [ ] 49% and 2.5 corrected; abstract and introduction tied to the tables
   - [ ] Kinetics-400 issue disclosed in one sentence, with no re-run promised
   - [ ] Memory misread corrected (host vs ≈5 MB)
   - [ ] Latency put in context (vs dense, vs host), with no speed-up claim
   - [ ] "Worst accuracy" corrected for 672 and conceded for 1024
   - [ ] Trade-off conceded (budget-matched point), claims narrowed
   - [ ] No deployment advantage claimed
   - [ ] Figure 1 caption, granularity table (appendix), formatting promised
   - [ ] No URLs; no new numbers outside the paper

---

## Sources

- AAAI-26 FAQ for Authors (Rebuttal): https://aaai.org/conference/aaai/aaai-26/aaai-26-faq-for-authors-rebuttal/
- AAAI-27 Review Process: https://aaai.org/conference/aaai/aaai-27/review-process/
- AAAI-27 Main Technical Track Call: https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/
- AAAI-26 Review Process: https://aaai.org/conference/aaai/aaai-26/review-process/
- Parikh, Batra & Lee, "How we write rebuttals": https://deviparikh.substack.com/p/how-we-write-rebuttals-dc84742fece1
- Gao et al., "Does My Rebuttal Matter? Insights from a Major NLP Conference", NAACL 2019: https://aclanthology.org/N19-1129/
- NeurIPS 2025 AC Guidelines: https://neurips.cc/Conferences/2025/AC-Guidelines
- NeurIPS 2023 Reviewer Guidelines: https://neurips.cc/Conferences/2023/ReviewerGuidelines
- CVPR 2025 Reviewer Guidelines and rebuttal template rules (no new contributions unless requested; one page): https://cvpr.thecvf.com/Conferences/2025/ReviewerGuidelines
