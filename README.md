# DoSRAGBench Prototype

**A proof-of-concept benchmark for denial-of-service attacks on Retrieval-Augmented Generation systems across the safety alignment spectrum.**

This prototype validates the thesis hypothesis that safety-aligned LLMs are paradoxically *more* vulnerable to denial-of-service than their base counterparts. It implements two attack categories (A1: Semantic Jamming, C1: Algorithmic Complexity) against matched base/instruct model pairs and produces real Alignment Vulnerability Index (AVI) numbers.

## What's Implemented

- **Category A1: Guardrail Triggering** — injects safety-trigger content into benign documents. Expected AVI >> 1 (alignment paradox).
- **Category C1: Embedding Space Clustering** — adversarial documents clustered near the query in HNSW embedding space to degrade retrieval latency. Expected AVI ≈ 1 (alignment-independent).
- **Six-metric framework** — ASR, GDS, LIR, TOR, CDR, plus AVI for aligned-vs-base comparison.
- **Refusal classifier** — pattern-based, tested with 18 unit tests covering safety refusal, epistemic refusal, hedged non-answers, generation failures.
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

### 3. Prepare data

```bash
python scripts/prepare_data.py --num-queries 50 --kb-size 1000
```

This downloads a Natural Questions subset (or falls back to synthetic data if HuggingFace is unreachable).

### 4. Run your first real experiment

```bash
# Requires ~12GB VRAM (RTX 4070 with 4-bit quantization)
python scripts/run_attack.py --category A1 --model-pair llama-3.1-8b --num-queries 20
python scripts/run_attack.py --category C1 --model-pair llama-3.1-8b --num-queries 20
```

### 5. Generate the alignment paradox report

```bash
python scripts/compute_avi.py
```

This produces a table like:

```
┌────────┬──────────┬─────────────┬───────────┬────────────────────────┐
│ Attack │ Base ASR │ Aligned ASR │ AVI (ASR) │ Interpretation         │
├────────┼──────────┼─────────────┼───────────┼────────────────────────┤
│ A1     │   5.0%   │    47.0%    │   9.40    │ Strong paradox         │
│ C1     │  12.0%   │    14.0%    │   1.17    │ Alignment-independent  │
└────────┴──────────┴─────────────┴───────────┴────────────────────────┘
```

Plus `results/avi_report.md` for inclusion in your seminar/thesis.

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

## Extending for Thesis B

To add a new attack category (e.g., B1 Context Saturation or D1 Logic Traps):

1. Create `src/dosragbench/attacks/b1_saturation.py` subclassing `DoSAttack`
2. Implement `generate_adversarial_docs(query, clean_docs)`
3. Register in `src/dosragbench/attacks/__init__.py` → `ATTACK_REGISTRY`
4. Add config to `configs/attacks.yaml`

The experiment runner, metrics, and AVI reporter require no changes.

## Known Limitations (Prototype Scope)

- **HNSW rebuild after attack:** FAISS HNSW doesn't support deletion, so we rebuild the index per query. For Thesis B, switch to a deletion-friendly backend (e.g., Weaviate with tombstones) for faster cycles.
- **Single embedder:** Currently uses `all-MiniLM-L6-v2`. Thesis B should test multiple embedders to validate embedder-independence.
- **No defence evaluation:** Phase 4 (perplexity filtering, NLI detection) not yet implemented — this is Thesis B scope.
- **Adversarial optimization is template-based:** C1 uses sampling + similarity filtering, not gradient-based optimization. For grey-box attacks against specific embedders, gradient methods should give tighter clusters (Thesis B).

## Citing

This prototype validates the framework proposed in the Thesis A proposal. When you have real numbers, update the "Expected Output" in this README with your actual AVI values — that's your hypothesis being proved/disproved.
