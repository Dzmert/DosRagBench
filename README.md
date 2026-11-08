# DoSRAGBench Prototype

**A benchmark for denial-of-service attacks on Retrieval-Augmented Generation systems across the instruction-tuning spectrum.**

Instruction-tuned LLMs are measurably *more* vulnerable to being silenced by adversarial documents than their base counterparts — 39 genuine paradoxes across 62 runs, under FDR-corrected Fisher plus McNemar. The mechanism is **context-faithfulness training**, not safety guardrails: models taught to answer only from the retrieved passages can be denied by making the evidence merely *look* inadequate.

Genuine safety refusals are 0.02% of aligned responses (22 of 100,000 on NQ). Epistemic refusals — "the context does not support an answer" — are 30%. See [`docs/reframing.md`](docs/reframing.md) for the evidence behind that claim and what it replaced.

## What's Implemented

- **Category A1: Context-Adequacy Attack** — injects content that makes the retrieved evidence read as unreliable or contested. (Named "Guardrail Triggering" until 2026-08-03; it turned out not to work through guardrails — the attack adds 5 safety refusals and 544 epistemic ones per 10,050 responses.)
- **Category C1: Embedding Space Clustering** — adversarial documents clustered near the query in HNSW embedding space. It evicts gold passages efficiently (1% corpus pollution already evicts ~60%, versus 1.5% for random injection at the same budget) and is a genuine paradox on three model families. It does **not** degrade retrieval latency: retrieval gets roughly twice as *fast* under attack, because greedy HNSW descent converges sooner once a dense adversarial cluster sits near the query.
- **Six-metric framework** — ASR, GDS, LIR, TOR, CDR, plus AVI. Note that AVI floors at ε = 0.01 on base ASR, so **risk difference is the primary effect size** and AVI is reported as a regime indicator; 21 of the 39 genuine runs are floored.
- **Refusal classifier** — pattern-based, 29 unit tests including regression tests for three fixed defects, and validated against 300 hand-labelled responses (holdout kappa 0.884; 0.725 against an independent annotator, which exceeds the 0.674 the two humans achieve with each other).
- **FAISS HNSW retriever** — matches production vector DBs (Pinecone, Weaviate, Milvus, Qdrant all use HNSW).
- **Matched model pair support** — Llama 3.1 8B base vs instruct by default; Qwen, Mistral, and DeepSeek-R1 pairs also configured.
- **Local + Katana HPC** — 4-bit quantization for RTX 4070; SLURM template included for 70B models.

## Quick Start

### 1. Install

```bash
cd dosragbench
pip install -e .
```

### 2. Validate the pipeline (no GPU required)

```bash
python scripts/smoke_test.py
```

Expected output:
```
A1 (Guardrail Triggering): AVI = 5.00 -- Strong paradox
C1 (Embedding Clustering): AVI = 1.00 -- Alignment-independent
✅ Smoke test PASSED
```

### Live demo

One question, both models, before and after the attack — the whole result in
about 30 seconds of compute:

```bash
python scripts/demo.py                 # llama-3.1-8b, attack D2
python scripts/demo.py --no-llm        # retrieval only, no GPU required
```

Models and the FAISS index load once and stay resident, so start it before you
need it. `:help` at the prompt lists the commands; `--no-llm` needs no GPU.

### 3. Prepare data

A small local build, for checking the pipeline runs end to end:

```bash
python scripts/prepare_data.py --num-queries 50 --kb-size 1000
```

**The published results do not use that.** Every reported run is built from a BEIR
corpus at ~500k passages:

```bash
python scripts/prepare_data.py --corpus beir --dataset {nq|hotpotqa} \
    --num-queries 1000 --kb-size 500000
```

NQ materialises 501,000 passages (1,000 gold + 500,000 filler) from a 2,681,468
passage corpus; HotpotQA materialises 500,995 (five gold ids are missing from the
corpus and are dropped). Seed 42. Filler is sampled rather than taken as a corpus
prefix, because BEIR corpora are not randomly ordered and a prefix would bias the
embedding neighbourhood the C-family attacks operate on.

### 4. Run your first real experiment

```bash
# Requires ~12GB VRAM (RTX 4070 with 4-bit quantization)
python scripts/run_attack.py --category A1 --model-pair llama-3.1-8b --num-queries 20
python scripts/run_attack.py --category C1 --model-pair llama-3.1-8b --num-queries 20
```

Reported runs use `--num-queries 1000`. Note that `configs/attacks.yaml` carries
`num_queries: 1000`, but the command line still wins if you pass the flag.

### 5. Generate the alignment paradox report

```bash
python scripts/recompute_metrics.py    # writes metrics_v2.json per run
python scripts/compute_significance.py # verdicts + figures
python scripts/compute_avi.py && python scripts/clean_avi_report.py
```

`clean_avi_report.py` writes the canonical `results/avi_report.md` — it applies the
minimum-answerable filter. `compute_avi.py` writes `avi_report_raw.*` and is
deliberately unfiltered, so the two disagree on run counts by design.

**Headline results** (62 retained runs, NQ 50 + HotpotQA 12), against a refusal
classifier validated on 300 hand-labelled responses (holdout kappa 0.884):

| | value |
|---|---|
| Genuine paradoxes | **39 of 62** |
| Mean risk difference, NQ excl. `llama-r1` | +0.189 (37/37 positive) |
| Mean risk difference, HotpotQA | +0.449 (12/12 positive) |
| `llama-r1-8b` (reasoning distillation) | −0.122, **11/13 protective** |

A run counts as a genuine paradox only if it passes **both** FDR-corrected Fisher
and within-model McNemar. See [`docs/findings_summary.md`](docs/findings_summary.md)
for the full result set and an explicit list of superseded numbers not to quote.

## Project Structure

```
dosragbench/
├── src/dosragbench/
│   ├── attacks/
│   │   ├── base.py              # Abstract DoSAttack class
│   │   ├── a1_guardrail.py      # MutedRAG-style A1
│   │   └── c1_clustering.py     # Novel embedding-space clustering
│   ├── metrics/
│   │   ├── refusal.py           # Refusal type + severity classifier
│   │   └── metrics.py           # ASR, GDS, LIR, TOR, CDR, AVI
│   ├── models/
│   │   └── loader.py            # HF model loading (4-bit local / HPC)
│   ├── pipeline/
│   │   ├── retriever.py         # FAISS HNSW retriever (timed)
│   │   └── rag.py               # End-to-end pipeline
│   └── utils/
│       └── config.py            # YAML config loading
├── configs/
│   ├── model_pairs.yaml         # Matched base/instruct pairs
│   └── attacks.yaml             # Attack parameters per category
├── scripts/
│   ├── prepare_data.py          # Download NQ subset, build KB
│   ├── run_attack.py            # Main experiment runner
│   ├── compute_avi.py           # Generate alignment paradox report
│   ├── smoke_test.py            # End-to-end test without GPU
│   └── submit_katana.sh         # SLURM template for HPC
├── tests/
│   └── test_refusal.py          # 18 unit tests (all passing)
├── data/                        # queries.json, knowledge_base.json
├── results/                     # Output: metrics.json, avi_report.md
└── pyproject.toml
```

## Model Pairs Available

Defined in `configs/model_pairs.yaml`:

| Pair Name     | Base                       | Aligned                          | Alignment Level |
|---------------|----------------------------|----------------------------------|-----------------|
| llama-3.1-8b  | Llama 3.1 8B Base          | Llama 3.1 8B Instruct            | 0 → 2           |
| qwen-2.5-7b   | Qwen 2.5 7B Base           | Qwen 2.5 7B Instruct             | 0 → 2           |
| mistral-7b    | Mistral 7B Base            | Mistral 7B Instruct              | 0 → 1           |
| llama-r1-8b   | Llama 3.1 8B Instruct      | DeepSeek-R1 Distill Llama 8B     | 2 → 4           |

The last pair is the reasoning comparison: same architecture, different post-training. Useful for testing Category D (CoT deadlocks) in Thesis B.

## Hardware Requirements

- **Local smoke test:** Any machine (uses stub models)
- **Local real experiments:** RTX 4070 or better (~12GB VRAM with 4-bit quantization)
- **Katana HPC:** Required for 70B+ models. Use `scripts/submit_katana.sh` as a template.

## Running on Katana

```bash
# On Katana login node, after cloning:
module load python/3.11 cuda/12.1
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Submit a job:
qsub -v PAIR=llama-3.1-8b,CATEGORY=A1,NUM_QUERIES=50 scripts/submit_katana.sh
```

## Running Tests

```bash
pytest tests/ -v
```

## Adding an attack

All 13 attacks in the taxonomy (A1–A3, B1–B3, C1–C3, D1–D4) are implemented and
have run. To add a fourteenth:

1. Create `src/dosragbench/attacks/<name>.py` subclassing `DoSAttack`
2. Implement `generate_adversarial_docs(query, clean_docs)`
3. Register in `src/dosragbench/attacks/__init__.py` → `ATTACK_REGISTRY`
4. Add config to `configs/attacks.yaml`

The experiment runner, metrics, and AVI reporter require no changes.

## Known limitations

- **No defence evaluation.** The Blocker baseline (Shafran et al.) is implemented
  in `attacks/blocker.py` but has not been run as a defence experiment. This is
  the largest gap.
- **HotpotQA is 12 of 52 cells.** A targeted probe, not a replication.
- **Base and aligned models receive different prompts.** `rag.py` selects the
  template on `chat_template`, which is exactly what distinguishes the two sides,
  so the comparison carries a prompt-wording difference. A `--prompt-style` flag
  now decouples the two and the ablation is pre-registered in
  [`docs/prompt_confound_preregistration.md`](docs/prompt_confound_preregistration.md);
  the runs are outstanding.
- **No answer-correctness ground truth.** BEIR supplies qrels but no answer
  strings, so every number measures whether the model *declined*, never whether
  it was right.
- **Single embedder** (`all-MiniLM-L6-v2`) and a **single seed per condition**.
  Generation is greedy and deterministic; the one run-to-run estimate available
  is 1.5 pp, on the C1 5% condition.
- **HNSW rebuild after attack:** FAISS HNSW doesn't support deletion, so the index
  is rebuilt per query. A deletion-friendly backend would cut cycle time.
- **C1 optimisation is template-based** — sampling plus similarity filtering, not
  gradient-based. Gradient methods should give tighter clusters against a known
  embedder.

## Citing

This prototype validates the framework proposed in the Thesis A proposal. When you have real numbers, update the "Expected Output" in this README with your actual AVI values — that's your hypothesis being proved/disproved.
