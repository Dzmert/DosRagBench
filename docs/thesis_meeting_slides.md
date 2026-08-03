# DoSRAGBench — Thesis Meeting Slide Outline

*Paste each block into PowerPoint / Google Slides. `TITLE` = slide title, bullets =
on-slide content (keep them short — the detail lives in the notes), `NOTES` = what
you say. Target length: ~10 slides + 3 backup, ~15 min.*

**Rebuilt 4 August 2026** against the validated classifier. Every number traces to
`results/avi_significance.json`, `results/avi_report_clean.json` or
`results/retrieval_binning.csv`, and each was recomputed for this revision.
Cross-check against `docs/findings_summary.md` §8 before presenting — it carries
the same figures plus an explicit do-not-quote list.

⚠️ **Two claims in the previous deck were wrong and are corrected here**: the
headline was 12 genuine of 50 (now **39 of 62**), and Slide 6 said the effect was
epistemic-only with retrieval attacks alignment-neutral (**false** — C-attacks are
genuine on three families, and nothing is alignment-independent). Both came from a
refusal classifier that has since been repaired and validated.

---

## Slide 1 — Title

**TITLE:** Trained to Defer: How Context-Faithfulness Creates a Denial-of-Service
Vulnerability in RAG

- DoSRAGBench: a benchmark for attack-induced denial across the instruction-tuning spectrum
- [Your name] · Supervisor: Dr Bao · [date]

**NOTES:** One sentence: "I built a benchmark to test whether instruction-tuned
LLMs are *more* vulnerable to being silenced by adversarial documents than their
base counterparts — the answer is yes, and the mechanism is not the one I
expected."

If asked why the title changed: the original said *safety* alignment. The data
says otherwise — genuine safety refusals are 0.02% of aligned responses. What the
attacks exploit is the model's trained deference to its retrieved context. See
`docs/reframing.md`.

---

## Slide 2 — The question

**TITLE:** Hypothesis, and what I actually measured

- **Original hypothesis:** *safety* alignment makes models more prone to
  denial-of-service (they refuse / break where base models answer)
- **Revised, and this is the contribution:** the vulnerability is real, but the
  mechanism is **context-faithfulness training**, not safety guardrails. Aligned
  models are taught to answer only from retrieved passages; the attack makes the
  evidence *look* inadequate. Safety refusals: 0.02% of aligned responses.
- **Metric — AVI** (Alignment Vulnerability Index) = ASR(aligned) / ASR(base)
- **ASR = attack-attributable denial:** of queries the model answers *with no
  attack*, the fraction the attack pushes into full denial
- Denial here is **semantic** (refusal / breakdown), **not** latency exhaustion

**NOTES:** Flag the naming honestly up front: classic DoS is compute/latency, but
what these attacks actually cause is *answer denial*. My latency metrics barely
move (LIR/TOR mostly 1.0–1.3×) — the denial is the model refusing or looping, not
the server slowing down. That's a framing decision I want your input on (Slide 8).

---

## Slide 3 — Method

**TITLE:** Design

- **4 matched pairs:** Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B (base→instruct) +
  Llama-3.1-Instruct → DeepSeek-R1-distill
- **13 attacks** in 4 families: A (guardrail/authority), B (context/generation),
  C (retrieval/embedding), D (epistemic/logical)
- **Two datasets:** BEIR NQ (clean recall@5 0.76) and BEIR HotpotQA (0.65),
  ~500k passages each, 1000 queries per run · FAISS HNSW (production-like)
- **67 runs → 62 kept** (NQ 50 + HotpotQA 12; 5 dropped for zero gold recall —
  retrieval failure, not a judgement call)

**NOTES:** Emphasise the matched-pair design — it's what makes the base-vs-aligned
contrast clean. Two points to pre-empt: (1) the exclusion list is *identical*
before and after the classifier was repaired, because those 5 fail on retrieval,
so nobody can claim it was tuned to the result; (2) HotpotQA is 12 of 52 cells —
call it a targeted probe, not a replication, before anyone else does.

---

## Slide 4 — Headline

**TITLE:** Not every "paradox" is real — and that's the contribution

- **51 / 62** runs point in the paradox direction (aligned worse)
- **39** survive both FDR-corrected Fisher **and** a within-model attack test
  → **genuine paradoxes**
- Full partition of the 62: **39 genuine · 11 protective · 10 baseline-floor
  artifacts · 2 not significant · 0 attack-but-alignment-independent**
- **37 / 37 positive** on NQ excluding the R1 pair; **12 / 12** on HotpotQA.
  No run in either group goes the other way.
- Genuine paradoxes span **3 model families and both datasets**

*Figure: `results/verdict_breakdown.png`*

**NOTES:** This is the slide to lead on. The honesty *is* the selling point: I
didn't report raw AVIs, I pre-filtered my own artifacts. Two tests: Fisher exact
(does aligned deny a different fraction than base?) + McNemar (does the *attack*
break queries that worked clean, or is the model just refusal-prone?). A claim
only counts if it passes both — that conjunction is what catches all 10 floor
artifacts.

If asked why this is 39 and the last version said 12: the refusal classifier had
a defect that mis-scored epistemic refusals, and I found it by validating the
instrument against 300 hand-labelled responses. Volunteer this — it is the
strongest thing on the slide, not the weakest. The 11 protective runs are all one
pair, and that is Slide 6.

---

## Slide 5 — Poster child

**TITLE:** The cleanest result: circular reference chains on Qwen-2.5

- **Attack D2 — Circular Reference Chains**, Qwen-2.5-7B base→instruct, NQ
- Base denial **2.2%** → aligned **52.4%**  ·  risk difference **+50.2 pp**
- McNemar: attack breaks **343** clean-working queries, "fixes" **22** (c:b ≈ 16)
- FDR-corrected q = **1.0 × 10⁻¹³⁵**
- Base ASR is above the ε floor, so **AVI 23.5 here is a real ratio**, not a
  division artefact

*Figure: `results/forest_genuine.png` (all 39 genuine runs with Wilson CIs)*

**NOTES:** Chosen deliberately over the old D3-Llama poster child, which is still
genuine but has a much weaker paired signal (c=207 vs b=99). Walk through why this
one is convincing: CIs don't touch, and the paired test shows the *attack* is
doing it — 343 broken against 22 repaired — not a pre-existing refusal habit. And
because base ASR clears 1%, this is one of the 18 genuine runs where AVI is a
measured ratio rather than a floored one (Slide 7).

---

## Slide 6 — The pattern

**TITLE:** It is not attack-specific — it is retrieval-quality-specific

- Genuine paradoxes by family (NQ): **Llama-3.1 (12/13)**, **Qwen (11/13)**,
  **Mistral (9/11)** — across *all four* attack families, including the
  retrieval C-attacks
- ⚠️ **This corrects my last slide deck**, which said the effect was epistemic-only
  and that retrieval attacks were alignment-neutral. On the repaired classifier
  C1/C2 are genuine on three families, and A3 is genuine rather than independent.
  Nothing is alignment-independent any more: that category is now **empty**.
- **The real gradient is retrieval quality.** Binning every query by the gold rank
  it had *before* the attack:

  | clean gold rank | aligned clean denial | gap under attack |
  |---|---|---|
  | rank 0 | 0.212 | +0.275 |
  | rank 1–2 | 0.260 | +0.422 |
  | rank 3–4 | 0.399 | +0.519 |
  | absent | 0.637 | +0.679 |

- Base models stay flat near zero across the same bins (0.007 → 0.025)

*Figure: `results/retrieval_gradient.png` — put this one on the slide, not the table*

**NOTES:** This is the mechanistic story, and it is *stronger* than the one I had.
Retrieval sensitivity is an aligned-model property — the gap grows monotonically as
retrieval degrades, with no exceptions in any group, while base models don't move.
That is exactly what context-faithfulness training predicts and what a
safety-guardrail account does not.

Say the correction out loud rather than letting them find it. The previous
"epistemic attacks only" claim was an artefact of the broken classifier
undercounting epistemic refusals on the C-attacks. Losing it is a gain: I no
longer need to explain why one attack family behaves differently.

**Rank 0 is protective** — gold as the top hit roughly halves the gap. That is the
defence lead: precision at rank 1 matters more than recall@5.

---

## Slide 6b — The reversal (new; do not cut this one)

**TITLE:** Reasoning distillation *reduces* the vulnerability

- 4th pair is **Llama-3.1-8B-Instruct → DeepSeek-R1-Distill** — alignment level
  2 → 4, so **both sides are instruction-tuned**
- **11 of 13 runs protective**, 0 genuine, mean risk difference **−0.122**
- Strongest: D2, base side 61.7% vs R1 29.2%, q = **6.7 × 10⁻³⁶**
- Across retrieval bins, unconditional denial under attack:
  **Instruct 0.328 → 0.722** vs **R1-Distill 0.092 → 0.350**

**NOTES:** This is worth more than another confirming pair. It makes the thesis a
claim about *which kind* of alignment creates the vulnerability, not a foregone
conclusion that all alignment does. Reasoning distillation appears to reduce
context deference — and context deference is the vulnerability.

It is also a built-in control: both models here have chat templates and get the
identical prompt, so the prompt-wording confound (Slide 7) cannot explain it.

Do not pool this pair into the headline mean — it is a different comparison, and
pooling it is what drags NQ from +0.189 down to +0.108.

---

## Slide 7 — Rigor / confounds I controlled

**TITLE:** Why I trust the 39

- **The instrument is validated.** 300 hand-labelled responses, boundary-weighted:
  holdout **kappa 0.884**, corpus-level error **2.05%**. An independent annotator
  gives **0.725** — *higher than the two humans agree with each other* (0.674).
- **Baseline refusal floor:** every aligned model refuses **29–37%** of clean NQ
  queries (73–81% on HotpotQA) with no attack present. Naive ASR credits the
  attack for that → I condition on answerable queries **and** run McNemar. That
  conjunction demoted **10 runs** to floor artifacts.
- **Multiple comparisons:** Benjamini–Hochberg FDR across all 62 tests pooled.
- **Semantic vs latency denial:** disclosed, not hidden (Slide 2).
- **Known open confound:** base and aligned models get different prompts, and the
  chat one says *"if the context doesn't contain the answer, say so briefly."*
  Ablation is designed and pre-registered; runs go up when the cluster returns.

**NOTES:** Pre-empt the tough questions by showing you already asked them. Lead
with the validation — it is the difference between "I wrote 34 regexes" and "I
measured my instrument against humans and it beats them."

**If asked why AVI, have the answer ready.** AVI = ASR(aligned)/ASR(base) with
ε = 0.01, so any run with base ASR under 1% returns aligned_ASR × 100 rather than
a measured ratio — that is **21 of the 39** genuine runs. So AVI is reported as a
*regime indicator*, and **risk difference is the primary effect size**. This is not
cosmetic: on the genuine set the two orderings agree only weakly (Spearman 0.69,
3/39 runs in the same rank position). AVI ranks D3-Llama top on a +31.8 pp effect
while risk difference ranks D2-Qwen-HotpotQA top at +68.4 pp. Whichever I lead
with, I have to say which and why — and the honest answer is that AVI is
interpretable to a reader and unstable near the floor, so it accompanies the risk
difference rather than replacing it.

---

## Slide 8 — Decisions I need from you

**TITLE:** Where I want your steer

1. **The reframe.** Title moves from *safety alignment* to *context-faithfulness*.
   Same vulnerability class, same method — only the identified mechanism moved,
   and it moved because I validated the instrument. Do you accept it?
2. **Where the remaining GPU budget goes:** finish HotpotQA (40 runs, 12/52 now)
   or evaluate a defence (Blocker baseline, already implemented and unused)?
   **I lean defence** — incomplete HotpotQA is a limitation I can state honestly;
   a missing defence evaluation is a missing contribution.
3. **Primary effect size:** risk difference with AVI as a regime indicator (my
   recommendation, see Slide 7), or bootstrap CIs on AVI itself?
4. **15 minutes of your time:** label 50 responses as an expert third annotator.
   Breaks the tie on three disputed gold labels without me touching my own gold
   set, and gives a Fleiss kappa independent of any one person.

**NOTES:** Come with recommendations, not open questions — they are written into
the bullets above. Have an opinion; let them adjust it.

On (1), expect pushback that "safety alignment" was the approved topic. The answer
is that the vulnerability class and the method are unchanged; what changed is the
mechanism I can *evidence*, and refusing to change it would mean reporting a claim
the data contradicts.

---

## Slide 9 — Summary

**TITLE:** Takeaways

- Instruction tuning creates a **real, significant** denial vulnerability —
  **39/62 runs**, 37/37 positive on NQ excluding the R1 pair, 12/12 on HotpotQA
- The mechanism is **context-faithfulness, not safety**: genuine safety refusals
  are **28 responses in 248,000**. What the attacks exploit is trained deference
  to retrieved context.
- Evidence is a **gradient, not an average** — the gap grows monotonically as
  retrieval degrades, and reasoning distillation reverses it
- A **validated instrument** (kappa 0.884; beats human–human agreement) plus
  CIs + FDR + McNemar is a **methodological contribution**, not just a results table

**NOTES:** Close on the reframed thesis in one sentence — *"models trained to
answer only from their context can be silenced by making the context look
inadequate"* — and hand back to the decisions on Slide 8.

---

## Backup A — Full significance table

Drop in the table from `results/avi_significance.md` (or a screenshot). Have it
ready for "show me all the runs" / "which ones failed and why."

## Backup B — Where the effect *doesn't* appear

- **R1-distill: 11/13 protective** — Slide 6b
- **No compute DoS anywhere.** Median latency inflation 1.09 aligned / 1.19 base;
  the *base* models inflate more on every metric, which is the opposite of a
  resource-exhaustion story. For C1 the retrieval ratio is **below 1.0 at every
  pollution level** — retrieval gets *faster* under attack, because greedy HNSW
  descent converges sooner once a dense adversarial cluster sits near the query.
- **Weakest attacks:** B1 Context Saturation is genuine on Qwen only; D4 is
  genuine on 3 of 4 pairs but with small effects (+0.025 to +0.042). Every
  designed attack does *something* on at least one pair — the honest negatives are
  the two above, not a failed attack.

## Backup C — The retrieval-layer result (C1)

- **1% corpus pollution already evicts ~60% of gold passages**; a 20× budget
  increase (5,010 → 100,200 docs) moves eviction by 5 pp. Flat at 60–65%, not a
  dose-response.
- **C1 vs a random-injection control at identical budget: 62.0% vs 1.5% eviction**
  — a 41× difference that rules out corpus-size inflation as the mechanism.
- Framing: *efficiency, not dose-response*. The attacker needs far less corpus
  access than a rising curve would imply — a stronger threat-model claim.

---

### Figures you already have
- `results/avi_summary.png` — two-panel heatmap (aligned ASR + AVI regime). Verified
  reproducible from raw data. **Caveat:** it plots *raw* AVI, so pair it with the
  Slide 4 breakdown or annotate the genuine cells so it doesn't oversell.
