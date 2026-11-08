# Pre-registration — bounding the prompt confound

**Written 3 August 2026, before any arm was run.** The predictions in §4 are
recorded in advance precisely so that the result cannot be read as post-hoc. If a
prediction here turns out wrong, the honest move is to say so in the thesis and
report the number, not to revise this file. Commit it before submitting any job.

Addresses the weakness logged as `findings_summary.md` §6.1, which is the most
attackable point in the methodology and aims directly at the dependent variable.

---

## 1. What the confound actually is

Base and aligned models do not receive the same prompt. `rag.py` selects
`RAG_PROMPT_CHAT` for models with a chat template and `RAG_PROMPT_BASE` for those
without — and chat-template availability is precisely what distinguishes base from
instruct. So every alignment comparison in the benchmark carries a prompt-wording
difference alongside the alignment difference.

Three differences, not one:

| | chat prompt (`rag.py:15`) | base prompt (`rag.py:38`) |
|---|---|---|
| faithfulness instruction | "using only the information from the context" | "using only the context" — **controlled, both have it** |
| **refusal licence** | **"If the context doesn't contain the answer, say so briefly."** | absent |
| brevity instruction | absent | "Keep answers short." |
| **exemplars** | zero-shot | **2 few-shot, both short direct answers** |

**The reframe raises the stakes.** If the mechanism is context-faithfulness, then a
prompt that says *"if the context doesn't contain the answer, say so briefly"* is a
direct instruction to exhibit the measured behaviour. An examiner who spots this
before we do can argue the entire result is a prompt artefact.

Note also that the confound is **bidirectional**: the chat prompt licenses refusal
*and* the base prompt demonstrates answering twice over. Both push the same way. A
single "aligned model with the base prompt" run therefore conflates removing a
refusal licence with adding two worked examples, which is why §3 uses four arms.

## 2. Why `chat_template` could not simply be flipped

That flag drives two separate things: prompt selection in `rag.py`, and
chat-template wrapping in `loader.py:38`. Setting `chat_template: false` on an
aligned model would change the prompt wording **and** strip the chat wrapping —
two variables at once, and a different experiment from the one that answers the
objection.

Resolved by adding an independent `prompt_style` parameter to `RAGPipeline`
(`auto` | `chat` | `chat-no-refusal` | `base`), exposed as `--prompt-style` on
`scripts/run_attack.py`. `auto` is the default and reproduces every run recorded
before 2026-08-03 exactly. `chat_template` is untouched, so chat wrapping is held
constant across all arms.

`--prompt-style` requires an explicit `--results-root`; the script refuses
otherwise. Ablation output goes to `results_promptablation/`, which the three
collectors do not scan, so **the 62-run headline grid cannot be perturbed by this
experiment.**

## 3. Design

`llama-3.1-8b`, attack **D3 (Epistemic Uncertainty)**, 1,000 queries, greedy
decoding. That pair has the highest aligned clean floor on NQ and is 12/13 genuine,
so it has the most to explain; D3 is the attack the reframe leans on most directly.

| arm | side | `--prompt-style` | isolates | status |
|---|---|---|---|---|
| **A0** | aligned | `auto` (= chat) | baseline | **already have** `results/llama-3.1-8b_D3` |
| **A1** | aligned | `chat-no-refusal` | the refusal licence alone | to run, ~2 h |
| **A2** | aligned | `base` | full prompt swap — the bound §6.1 asks for | to run, ~2 h |
| **A3** | **base** | `chat` | reverse direction | to run, ~2 h |

`chat-no-refusal` is `RAG_PROMPT_CHAT` with exactly the sentence *"If the context
doesn't contain the answer, say so briefly."* deleted and nothing else altered —
asserted in code, not by eye. It deliberately **keeps** the faithfulness
instruction, so A0 → A1 separates *permission to decline* from *answer only from
the context*. The second is the claimed mechanism; the first is a prompt artefact.

**A3 is the arm to protect if budget is cut.** It asks the question directly: give
the *base* model the refusal licence, and does it start refusing? That is a cleaner
rebuttal than any bound derived from the aligned side, because it cannot be
explained away by anything specific to instruction tuning.

### Commands

```bash
for style_side in "chat-no-refusal aligned" "base aligned" "chat base"; do
  set -- $style_side
  python3 scripts/run_attack.py --category D3 --model-pair llama-3.1-8b \
      --num-queries 1000 --side $2 --prompt-style $1 \
      --results-root results_promptablation
done
```

Each writes to `results_promptablation/llama-3.1-8b_D3_prompt-<style>_<side>/`
with a `run_config.json` recording the arm.

## 4. Predictions, recorded in advance

A0 measured values, from `avi_report_clean.json` and `avi_significance.json`:

| quantity | A0 value |
|---|---|
| aligned clean-denial floor | **0.371** |
| base clean-denial floor | **0.004** |
| aligned attributable ASR | 0.329 |
| base attributable ASR | 0.011 |
| risk difference | **+0.318** (GENUINE, q = 1.2e-79) |
| aligned answerable denominator | 629 |

**Primary endpoint: the aligned clean-denial floor in A2.** It is measured without
any attack, so it is the cleanest read on what the prompt alone buys.

| outcome | A2 aligned floor | A3 base floor | reading |
|---|---|---|---|
| **confound explains the effect** | falls below ~0.10 | rises above ~0.20 | the gap is a prompt artefact; the thesis needs restructuring around a prompt-controlled design |
| **mechanism is real** | stays above ~0.20 | stays below ~0.05 | context-faithfulness is trained in, not prompted; §6.1 downgrades from a threat to a bounded caveat |
| **partial** | 0.10–0.20 | 0.05–0.20 | report the split explicitly: A0 − A1 is the licence, A1 − A2 is the exemplars, the remainder is training |

**I predict the middle row**, with the A2 floor landing between 0.20 and 0.30 and
A3 staying under 0.05. Reasoning: the base models sit at 0.004 while *carrying the
same faithfulness instruction* the aligned models get, which already suggests the
instruction is not sufficient to produce refusal; and §3.7's binning shows aligned
clean denial rising monotonically with retrieval degradation (0.212 → 0.637) under
a **fixed** prompt, which no wording difference can explain.

Secondary expectations, recorded but not primary:

1. **A1 sits between A0 and A2.** If A1 ≈ A0, the refusal licence does nothing and
   the few-shot exemplars carry the whole prompt effect — worth knowing, and it
   would mean the sentence everyone objects to is not the problem.
2. **Risk difference survives in A2**, at reduced magnitude. If it inverts, that is
   a much bigger finding than the confound and supersedes this document.
3. **The A2 answerable denominator rises** above 629 as the floor drops, which
   mechanically shrinks the selection artifact described in §6.2. Report
   `num_answerable` for every arm.

### Falsification

This experiment refutes the thesis's central mechanism claim if **A2's aligned
clean floor falls below 0.10 while A3's base floor rises above 0.20**. That is the
result that would mean the alignment gap is substantially a prompt artefact. It is
stated here so that it cannot be quietly renegotiated afterwards.

## 5. What was already immune before running

Two results are unaffected by prompt wording by construction, and should be cited
whenever this objection is raised regardless of how the arms land:

1. **The retrieval-quality binning (§3.7)** is a *within-aligned-model*
   dose-response: the same model, the same prompt, denial rising monotonically as
   retrieval degrades. A fixed prompt cannot produce a gradient.
2. **The `llama-r1` reversal (§3.4)** compares two models that **both** have chat
   templates and therefore both receive the identical chat prompt — and it shows a
   large effect in the *opposite* direction (11/13 protective, q → 6.7e-36).

Neither depends on the base/aligned prompt difference at all. The arms in §3
tighten the bound; they are not what the mechanism claim rests on.

## 6. Scoring

```bash
python3 scripts/recompute_metrics.py --results-dir results_promptablation
python3 scripts/compute_significance.py --results-dir results_promptablation \
    --out results_promptablation/significance.md
```

Then rewrite `findings_summary.md` §6.1 from "disclose it" to a measured quantity,
and add the primary endpoint to the §8 quotable-numbers table.

Do **not** pass `results_promptablation/` to the collectors alongside `results/`.
These arms are not part of the 62-run grid and pooling them would change the
headline counts.
