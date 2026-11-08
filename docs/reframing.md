# The mechanism is context-faithfulness, not safety

*Decided 2026-08-03, on evidence from 62 validated runs. This document states the
revised claim, the evidence for it, and what changes in the writeup. The full
chronology of how it was discovered is in `HANDOFF.md`.*

## The claim that survives

Instruction-tuned LLMs are more vulnerable to attack-induced denial than their
base counterparts. **39 of 62 runs are genuine paradoxes** — passing both an
FDR-corrected Fisher test (aligned denies a different fraction than base) and a
McNemar test (the attack actually breaks queries that worked clean).

Nothing about that is retracted. The benchmark, the 13 attacks, the six-metric
framework, the attributable-ASR definition and the Fisher+McNemar protocol all
stand unchanged.

## The claim that does not survive

> *Safety* alignment creates the vulnerability.

Genuine safety refusals are **0.02%** of aligned responses (22 of 100,000 on NQ;
19 of 24,000 on HotpotQA). They are not the mechanism, and they are not
statistically capable of being the mechanism.

What the attacks exploit is **context-faithfulness training** — the instruction
that a RAG model should answer only from its retrieved passages. That is a
defence against hallucination, and it is standard practice. The attack does not
need to trigger a guardrail; it only needs to make the retrieved evidence *look*
inadequate.

## Three independent lines of evidence

### 1. Composition — the refusals are epistemic, not safety

| dataset / side | epistemic | explicit_safety |
|---|---|---|
| NQ aligned | 30.2% | **0.02%** (22 / 100,000) |
| NQ base | 11.2% | 0.006% (6) |
| HotpotQA aligned | 79.6% | **0.08%** (19 / 24,000) |
| HotpotQA base | 2.9% | 0.00% (0) |

Aligned models refuse 3× more than base on NQ and 27× more on HotpotQA, almost
entirely with "the context does not support an answer".

### 2. Dose-response — the effect tracks retrieval quality

Binning every query on the `gold_rank` it had *before* the attack
(`scripts/retrieval_binning.py`):

| clean gold rank | aligned clean denial | uncond. gap under attack |
|---|---|---|
| rank 0 | 0.212 | +0.275 |
| rank 1–2 | 0.260 | +0.422 |
| rank 3–4 | 0.399 | +0.519 |
| absent | 0.637 | +0.679 |

Monotonic in every group, on both datasets. Base models stay flat near zero
(0.007 → 0.021). **Safety guardrails have no reason to care where the gold
document ranked; context-faithfulness cares about exactly that.**

### 3. The counterexample runs the wrong way for a safety account

`DeepSeek-R1-Distill-Llama-8B` sits at `alignment_level=4`, above
`Llama-3.1-8B-Instruct` at 2. A safety account predicts it is the *most*
vulnerable pair. It is **protective in 11 of 13 runs**, at q down to 6.7e-36.

The binning explains why: R1-Distill is far less context-deferent
(0.092 → 0.350 across retrieval bins, against Instruct's 0.328 → 0.722).
Reasoning distillation reduces deference to the retrieved context, and deference
is the vulnerability. The counterexample is *explained by* the revised mechanism
rather than excused as an anomaly.

### 4. The purpose-built test already came back negative

The A-family attacks were designed as "Guardrail Triggering". Aligned side,
clean → attacked:

| family | explicit_safety | epistemic |
|---|---|---|
| A (guardrail-triggering) | 2 → **7** / 10,050 | 4,316 → **4,860** |
| D | 1 → **18** / 16,000 | 7,171 → **9,066** |

The attack built specifically to trigger guardrails adds **5 safety refusals and
544 epistemic ones**. The safety channel is under 1% of the effect. This is the
strongest available internal test of the original hypothesis and it is negative.

## What changes in the writeup

| | before | after |
|---|---|---|
| Title | "Does Safety Alignment Create a DoS Vulnerability in RAG?" | "Trained to Defer: How Context-Faithfulness Creates a DoS Vulnerability in RAG" |
| RQ1 | does *safety alignment* increase susceptibility | does *instruction tuning* increase susceptibility |
| RQ2 | which attack mechanisms drive it | which *trained behaviour* drives it — guardrails or context-faithfulness |
| §2.2 | "Safety alignment" — RLHF/DPO, refusal | "Alignment and groundedness" — split safety refusal from faithfulness training; cite the groundedness literature |
| A-family | "Guardrail Triggering" | "Context-Adequacy Attack" |
| Discussion | alignment tax on safety tuning | alignment tax on *groundedness* tuning — the hallucination defence is the availability vulnerability |

## Why this is the stronger thesis

- **Mechanistic rather than correlational.** It names a specific trained
  behaviour and predicts where the effect will be larger.
- **It predicted the dose-response** that §2 then found.
- **It explains the counterexample** instead of setting it aside.
- **It yields a deployable recommendation:** precision at rank 1 matters more
  than recall@5 for DoS resistance.
- **It generalises further.** Groundedness training is near-universal in
  production RAG, and is applied to models with no safety tuning at all. The
  vulnerability is correspondingly wider than a safety story would imply.

## The tension worth stating explicitly

Context-faithfulness training exists to prevent hallucination. It works. The
finding is that the same mechanism converts an availability attack into a cheap
one — a direct trade-off between two desirable properties, not a bug in either.
That framing is worth a paragraph in the discussion; it is the most quotable
thing in the thesis.

## Open

A domain where safety training genuinely fires (medical, legal, harm-adjacent)
has not been tested. If safety refusals appear at meaningful rates there and
attacks amplify them, the thesis becomes "two channels, domain-dependent" rather
than one. That is a discriminating experiment worth running — either outcome is
publishable — but it does not change the NQ/HotpotQA conclusion, which is
epistemic regardless.
