# DoSRAGBench — Thesis Meeting Slide Outline

*Paste each block into PowerPoint / Google Slides. `TITLE` = slide title, bullets =
on-slide content (keep them short — the detail lives in the notes), `NOTES` = what
you say. Every number traces to `results/avi_significance.md` and
`results/avi_report.md`. Target length: ~9 slides + 2 backup, ~15 min.*

---

## Slide 1 — Title

**TITLE:** Does Safety Alignment Create a Denial-of-Service Vulnerability in RAG?

- DoSRAGBench: a benchmark for attack-induced denial across the alignment spectrum
- [Your name] · Supervisor: Dr Bao · [date]

**NOTES:** One sentence: "I built a benchmark to test whether safety-tuned LLMs
are *more* vulnerable to being silenced by adversarial documents than their base
counterparts — and the answer is a qualified yes, for a specific class of attack."

---

## Slide 2 — The question

**TITLE:** Hypothesis, and what I actually measured

- **Original hypothesis:** safety alignment makes models *more* prone to
  denial-of-service (they refuse / break where base models answer)
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
- ~1000 Natural Questions queries per run · FAISS HNSW retriever (production-like)
- **56 runs → 50 kept** (5 dropped: too few answerable queries to be reliable)

**NOTES:** Emphasise the matched-pair design — it's what makes AVI a clean
base-vs-aligned contrast. Mention the R1-distill pair is the interesting
"reasoning model" arm.

---

## Slide 4 — Headline

**TITLE:** Not every "paradox" is real — and that's the contribution

- **27 / 50** runs point in the paradox direction (aligned worse)
- Only **12** survive both FDR-corrected significance **and** a within-model
  attack test → **genuine paradoxes**
- Breakdown of the 27: **12 genuine · 5 baseline-floor artifacts · 2
  attack-but-alignment-independent · 8 not significant**
- The other 23 runs point the *other* way: **8 protective · 15 not significant**
- Genuine paradoxes span **3 model families**, not just one

*Figure: `results/verdict_breakdown.png`*

**NOTES:** This is the slide to lead on. The honesty *is* the selling point: I
didn't just report raw AVIs, I pre-filtered my own artifacts. Two tests: Fisher
exact (does aligned deny a different fraction than base?) + McNemar (does the
*attack* break queries that worked clean, vs the model just being refusal-prone?).
A claim only counts as a paradox if it passes both.

---

## Slide 5 — Poster child

**TITLE:** The cleanest result: epistemic attack on Llama-3.1

- **Attack D3 — Epistemic Uncertainty Amplification**, Llama-3.1 base→instruct
- Base denial **0.8%** [0.4–1.6] → aligned **8.4%** [6.9–10.3]  ·  **AVI 8.43**
- McNemar: attack breaks **84** clean-working queries, "fixes" **3**
  (p = 1.4 × 10⁻²¹)
- Non-overlapping CIs, huge attack effect — this one is bulletproof

*Figure: `results/forest_genuine.png` (all 12 genuine runs with Wilson CIs)*

**NOTES:** Walk through why this is convincing: the confidence intervals don't
touch, and the paired test shows the *attack* is doing it (84 vs 3), not a
pre-existing refusal habit. Also note: this is Llama, not Qwen — so the paradox
isn't a single-model quirk.

---

## Slide 6 — The pattern

**TITLE:** Alignment is vulnerable to *epistemic* attacks, not retrieval attacks

- Genuine paradoxes by family: **Qwen (8)**, **Llama-3.1 (B2, D1, D3)**,
  **Mistral (D2)**
- Every non-Qwen genuine paradox is a **reasoning/epistemic attack** (B2
  generation-loop, D-family logical/circular/epistemic) — **none** are the
  retrieval C-attacks
- **Built-in controls that validate the frame:**
  - **A3 Authority Spoofing** — real attack, but *alignment-independent* (breaks
    base & aligned equally: 48/0, 60/0; Fisher n.s.)
  - **R1-distill** model — *protective* almost everywhere (denials drop to ~0)

**NOTES:** This is the mechanistic story that turns a pile of numbers into a
thesis: alignment tuning seems to make models fold under *epistemic* pressure
(uncertainty, contradiction, circular reasoning), while retrieval-space attacks
are alignment-neutral. A3 and the R1 result are effectively my control conditions
— they show the effect is specific, not an artifact of my pipeline.

---

## Slide 7 — Rigor / confounds I controlled

**TITLE:** Why I trust the 12

- **Baseline refusal floor:** aligned Qwen refuses ~17% of queries with *no
  attack*. Naive ASR credits the attack for that → I condition on answerable
  queries **and** run McNemar. This is exactly what demoted **5 Qwen "paradoxes"
  (A2, C3, D1, D3, D4)** to *floor artifacts*.
- **Near-zero base denominator:** when base ASR ≈ 0, AVI explodes (÷ε). I report
  these as "attack works on aligned only," with CIs — not as precise ratios.
- **Multiple comparisons:** Benjamini-Hochberg FDR across all 50 runs.
- **Semantic vs latency denial:** disclosed, not hidden (Slide 2).

**NOTES:** Pre-empt the tough questions by showing you already asked them. The Qwen
floor-artifact catch is the strongest evidence of rigor — I demoted my own
biggest-looking numbers because the within-model test didn't back them.

---

## Slide 8 — Decisions I need from you

**TITLE:** Where I want your steer

1. **Scope:** go *deep* (mechanism — why epistemic attacks? why Qwen worst?) or
   *wide* (more aligned models — paradox is in 3/4 families; is it general)?
2. **The "DoS" framing:** reframe as *induced-refusal / availability* (matches the
   data), or add genuine compute-DoS attacks to earn the name?
3. **Stats bar:** is FDR + McNemar the standard you want, or add bootstrap CIs on
   AVI itself before I write this up?
4. **Target:** what venue / thesis-chapter framing are we aiming at?

**NOTES:** Come with a recommendation, not open questions: e.g. "I lean toward
scope = mechanism on the epistemic attacks, and reframing as availability rather
than adding compute-DoS work — but I want your call." Have an opinion; let them
adjust it.

---

## Slide 9 — Summary

**TITLE:** Takeaways

- Safety alignment creates a **real, significant** denial vulnerability — but only
  to **epistemic/reasoning attacks**, and **smaller** than raw AVI suggests
- Effect is **cross-family** (Qwen, Llama, Mistral), with clean controls (A3
  independent; R1-distill protective)
- Rigorous filtering (CIs + FDR + McNemar) is a **methodological contribution**,
  not just a results table

**NOTES:** Close on the reframed thesis in one sentence and hand back to the
scope/naming decisions.

---

## Backup A — Full significance table

Drop in the table from `results/avi_significance.md` (or a screenshot). Have it
ready for "show me all the runs" / "which ones failed and why."

## Backup B — Where the effect *doesn't* appear

- Retrieval C-attacks: mostly protective or n.s. across non-Qwen families
- R1-distill: protective on 8, both-zero on 5 — reasoning distillation appears to
  reverse the paradox (a lead for future work)
- Latency/token tables (`avi_report.md`): attacks barely move compute cost —
  supports the "semantic denial" reframing

---

### Figures you already have
- `results/avi_summary.png` — two-panel heatmap (aligned ASR + AVI regime). Verified
  reproducible from raw data. **Caveat:** it plots *raw* AVI, so pair it with the
  Slide 4 breakdown or annotate the genuine cells so it doesn't oversell.
