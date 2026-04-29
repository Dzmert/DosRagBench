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

### 2. Validate the pipeline 
```bash
python scripts/smoke_test.py
```

Expected output:
```
A1 (Guardrail Triggering): AVI = 5.00 -- Strong paradox
C1 (Embedding Clustering): AVI = 1.00 -- Alignment-independent
Smoke test PASSED
```

### 3. Prepare data

```bash
python scripts/prepare_data.py --num-queries 50 --kb-size 1000
```

This downloads a Natural Questions subset (or falls back to synthetic data if HuggingFace is unreachable).

### 4. Run first experiment

```bash
python scripts/run_attack.py --category A1 --model-pair llama-3.1-8b --num-queries 20
python scripts/run_attack.py --category C1 --model-pair llama-3.1-8b --num-queries 20
```

### 5. Generate a report of the results

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



## Model Pairs Available

Defined in `configs/model_pairs.yaml`:

| Pair Name     | Base                       | Aligned                          | Alignment Level |
|---------------|----------------------------|----------------------------------|-----------------|
| llama-3.1-8b  | Llama 3.1 8B Base          | Llama 3.1 8B Instruct            | 0 → 2           |
| qwen-2.5-7b   | Qwen 2.5 7B Base           | Qwen 2.5 7B Instruct             | 0 → 2           |
| mistral-7b    | Mistral 7B Base            | Mistral 7B Instruct              | 0 → 1           |
| llama-r1-8b   | Llama 3.1 8B Instruct      | DeepSeek-R1 Distill Llama 8B     | 2 → 4           |

The last pair is the reasoning comparison: same architecture, different post-training. Useful for testing Category D (CoT deadlocks) in Thesis B.


## Running Tests

```bash
pytest tests/ -v
```



