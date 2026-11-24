# DoSRAGBench

**A benchmark for denial-of-service attacks on Retrieval-Augmented Generation
systems across the instruction-tuning spectrum.**

DoSRAGBench asks whether poisoned passages can make a RAG system *stop
answering*, and whether the instruction tuning that makes a model a usable
assistant also makes it easier to silence. It runs 13 attacks in four families
against four matched model pairs, on BEIR Natural Questions (501,000 passages)
and BEIR FiQA-2018 (57,638 passages).

## Headline results

Aligned models are consistently the more vulnerable side.

| | NQ | FiQA |
|---|---|---|
| Genuine paradoxes (FDR-corrected Fisher **and** McNemar) | **32 of 50** runs | **29 of 52** runs |
| Mean risk difference, true base↔aligned pairs | +0.189 (37/37 positive) | +0.209 (39/39 positive) |
| `llama-r1-8b` (reasoning distillation) | −0.122, 11/13 protective | −0.159, 11/13 protective |
| Clean recall@5 | 0.760 | 0.409 |

A run counts as a genuine paradox only if the aligned model is significantly
more vulnerable (Fisher, Benjamini–Hochberg FDR within each corpus) **and** the
attack itself breaks queries that worked clean (McNemar, c > b).
`llama-r1-8b` compares two instruction-tuned models (Llama-3.1-8B-Instruct vs
DeepSeek-R1-Distill), so it is reported separately and never pooled.

**The mechanism is context-faithfulness, not safety.** Only 28 of 248,000
responses are safety refusals. Almost every refusal is epistemic: "the context
does not support an answer". On NQ, aligned models decline 29–37% of queries
with no attack present; base models decline 0.1–1.3%.

**Prompt confound, measured.** Base and aligned models receive different prompt
templates. A pre-registered ablation attributes about 63% of the aligned clean
refusal floor to prompt wording. With wording held constant, the alignment
effect is **+0.134** on the clean floor and **+0.204** on attack success. The
same refusal instruction raises a base model's floor by 0.074 and an aligned
model's by 0.233 (a 3.1× difference). All other figures use `auto` prompts and
are upper bounds.

**Silent error versus visible refusal.** Scoring 656 NQ queries against
NQ-open short answers shows that attacks cost aligned models 6.9pp more of their
correct answers (37/37 runs). But 76–85% of the aligned model's losses become
refusals, while 95–97% of the base model's losses become wrong answers. The same
attack is a denial-of-service attack on the aligned model and a misinformation
attack on the base model.

**Defences.** A near-duplicate cluster filter on the top-50 candidates cuts
D2 / B3 / C1 attack success from 58% / 36% / 20% to 5% / 1% / 9% (Llama;
replicated on Qwen), with no measurable clean cost. Diversified contradiction
flooding (A2) evades it. Cross-encoder reranking improves clean retrieval but
roughly doubles A2's success (26% → 52%).

**Negative results.** No attack produces compute denial of service: median
latency inflation is 1.09× (aligned), and C1 makes retrieval *faster*. An
instruction-injection blocker (after Shafran et al.) reverses the paradox: the
base model is the more vulnerable one to instructions, the aligned model to
degraded evidence.

**Withdrawn.** A HotpotQA arm (12 runs) was withdrawn: the corpus builder kept
one gold passage per query, and every HotpotQA question needs two.

## Measurement

- **Refusal classifier.** Pattern-based, validated against 300 hand-labelled
  responses: holdout κ 0.884; κ 0.725 against an independent annotator (higher
  than the two humans' 0.674 with each other). Population-reweighted error
  2.05%, skewed toward missing refusals, so the measured gap is conservative.
- **Retrieval.** `all-MiniLM-L6-v2` embedder, FAISS HNSW (M 16, efSearch 50),
  top-5.
- **Generation.** Greedy decoding, 4-bit quantisation, all eight models.
- **Compute.** About 470 GPU-hours on UNSW Katana.

## Attacks

| Family | Attacks |
|---|---|
| A: context adequacy | A1 guardrail triggering (acts on context adequacy, not guardrails), A2 contradiction flooding, A3 authority spoofing |
| B: context and generation | B1 context saturation, B2 generation loop, B3 multi-retrieval amplification |
| C: retrieval layer | C1 embedding-space clustering, C2 index pollution, C3 adversarial embedding perturbation |
| D: reasoning traps | D1 logical contradiction, D2 circular reference, D3 epistemic uncertainty, D4 infinite qualification |

Strongest by McNemar ratio on NQ: D2 and C1 (7.2), B3 (6.8), A2 (5.2). C1 evicts
the gold passage for ~60% of queries at 1% corpus pollution, flat to 20%, against
1.5% for random injection (`RAND`) at the same budget.

External baselines (Shafran et al., arXiv:2406.05870) live in
`attacks/blocker.py`, `blocker_construct.py` and `jamming.py`.

## Quick start

### 1. Install

```bash
cd dosragbench
pip install -e .
```

### 2. Validate the pipeline (no GPU required)

```bash
python scripts/smoke_test.py
pytest tests/ -v
```

### Live demo

One question, both models, before and after the attack:

```bash
python scripts/demo.py                 # llama-3.1-8b, attack D2
python scripts/demo.py --no-llm        # retrieval only, no GPU required
```

Models and the FAISS index load once and stay resident. `:help` at the prompt
lists the commands.

### 3. Prepare data

A small local build, for checking the pipeline runs end to end:

```bash
python scripts/prepare_data.py --num-queries 50 --kb-size 1000
```

The reported runs use full BEIR builds:

```bash
python scripts/prepare_data.py --corpus beir --dataset nq \
    --num-queries 1000 --kb-size 500000
python scripts/prepare_data.py --corpus beir --dataset fiqa \n    --num-queries 1000 --kb-size 60000 --seed 42 --output-dir data_fiqa
```

NQ materialises 501,000 passages (1,000 gold + 500,000 sampled filler) from a
2,681,468-passage corpus, seed 42. Filler is sampled rather than taken as a
prefix, because BEIR corpora are not randomly ordered. FiQA's 57,638-passage corpus fits under that cap, so it is used whole, with
all 648 test queries. Screen any new dataset with `scripts/check_gold_multiplicity.py` first:
it reports whether multiple gold passages are redundant (usable) or
complementary (not usable with the single-gold builder).

`data/knowledge_base.json` is stored with Git LFS (`git lfs pull`).

### 4. Run an experiment

```bash
# ~12 GB VRAM with 4-bit quantisation
python scripts/run_attack.py --category D2 --model-pair llama-3.1-8b --num-queries 20
```

Reported runs use `--num-queries 1000` (NQ) or all 648 queries (FiQA).

### 5. Regenerate the results

```bash
python scripts/recompute_metrics.py
python scripts/compute_significance.py
python scripts/compute_significance.py --results-dir results_fiqa --out results_fiqa/avi_significance.md
python scripts/compute_avi.py && python scripts/clean_avi_report.py
python scripts/score_validation.py
python scripts/retrieval_binning.py
python scripts/score_correctness.py && python scripts/score_correctness.py --match lenient
python scripts/compare_defences.py
```

`score_correctness.py` needs `data/nq_gold_answers.json`, built once by
`scripts/fetch_nq_answers.py`. On Windows, set `PYTHONUTF8=1`.

The verdicts behind the headline table are in `results/avi_significance.json`
(NQ) and `results_fiqa/avi_significance.json` (FiQA).

## Model pairs

Defined in `configs/model_pairs.yaml`:

| Pair | "Base" side | "Aligned" side | Level |
|---|---|---|---|
| `llama-3.1-8b` | Llama 3.1 8B | Llama 3.1 8B Instruct | 0 → 2 |
| `qwen-2.5-7b` | Qwen 2.5 7B | Qwen 2.5 7B Instruct | 0 → 2 |
| `mistral-7b` | Mistral 7B v0.3 | Mistral 7B Instruct v0.3 | 0 → 1 |
| `llama-r1-8b` | Llama 3.1 8B Instruct | DeepSeek-R1-Distill-Llama-8B | 2 → 4 |

## Project structure

```
dosragbench/
├── src/dosragbench/
│   ├── attacks/       # 13 attacks (A1–D4), RAND control, blocker baselines
│   ├── metrics/       # refusal classifier, ASR / GDS / LIR / TOR / CDR / AVI
│   ├── models/        # HF model loading (4-bit)
│   ├── pipeline/      # FAISS HNSW retriever, RAG pipeline, defences
│   └── utils/
├── configs/           # model_pairs.yaml, attacks.yaml
├── scripts/           # data prep, experiment runner, analysis, Katana PBS jobs
├── tests/             # refusal classifier, correctness, blocker, defence tests
├── validation/        # hand-labelled classifier validation sheets
├── results/           # NQ runs
├── results_fiqa/      # FiQA runs
├── results_defence/   # defence experiments
└── results_promptablation/
```

## Running on Katana

Katana uses PBS. Job scripts are in `scripts/` (`submit_katana.sh`,
`run_*_katana.pbs`).

```bash
module load python/3.11 cuda/12.1
python -m venv .venv && source .venv/bin/activate
pip install -e .

qsub -v PAIR=llama-3.1-8b,CATEGORY=D2,NUM_QUERIES=1000 scripts/submit_katana.sh
```

## Limitations

- **Prompt confound.** Main-grid figures use different templates for base and
  aligned models, so they are upper bounds. The prompt-matched estimate comes
  from one pair, one attack and one corpus.
- **Correctness** is scored on NQ only (656 of 1,000 queries); FiQA has no short
  answers.
- **Defences** are tested on two pairs, NQ only, against a non-adaptive
  attacker. No single cluster-filter setting is best against both templated
  attacks and A2.
- **Scale.** 7–8B models, one embedder, one seed per condition (generation is
  deterministic; the one run-to-run retrieval estimate is 1.5pp).
- **Optimised blocker** ran at 0.6% of the original paper's search budget and is
  a null result at that budget only.
- **C1 is template-based**, not gradient-optimised.
