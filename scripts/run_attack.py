"""Run a DoS attack experiment on a matched base/aligned model pair.

This is the primary experiment script. For each query:
  1. Run baseline RAG (clean KB) on both base and aligned model
  2. Inject adversarial docs for that query
  3. Run attacked RAG on both base and aligned model
  4. Remove adversarial docs before the next query

Outputs per-query results + aggregated metrics to results/ for later AVI analysis.

Usage:
    python scripts/run_attack.py --category A1 --model-pair llama-3.1-8b
    python scripts/run_attack.py --category C1 --model-pair llama-3.1-8b --num-queries 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.attacks import build_attack
from dosragbench.metrics import QueryResult, compute_metrics
from dosragbench.models import detect_environment, load_model
from dosragbench.pipeline import Document, HNSWRetriever, RAGPipeline
from dosragbench.utils.config import (
    DATA_DIR,
    RESULTS_DIR,
    load_attack,
    load_model_pair,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_dataset() -> tuple[list[dict], list[Document]]:
    """Load queries and knowledge base from data/."""
    queries_path = DATA_DIR / "queries.json"
    kb_path = DATA_DIR / "knowledge_base.json"

    if not queries_path.exists() or not kb_path.exists():
        raise FileNotFoundError(
            "Missing data/queries.json or data/knowledge_base.json. "
            "Run scripts/prepare_data.py first."
        )

    with open(queries_path) as f:
        queries = json.load(f)
    with open(kb_path) as f:
        raw_docs = json.load(f)

    docs = [Document(doc_id=d["doc_id"], text=d["text"]) for d in raw_docs]
    return queries, docs


def run_model_side(
    model_config,
    retriever: HNSWRetriever,
    queries: list[dict],
    attack,
    top_k: int,
    num_queries: int,
) -> dict:
    """Run baseline + attacked RAG for a single model.

    Returns {'baseline': [QueryResult...], 'attacked': [QueryResult...]}.
    """
    logger.info(f"Loading model: {model_config.name}")
    loaded = load_model(model_config)
    rag = RAGPipeline(retriever=retriever, model=loaded, top_k=top_k)

    baseline_results: list[QueryResult] = []
    attacked_results: list[QueryResult] = []

    for i, q in enumerate(queries[:num_queries]):
        query_text = q["query"]
        logger.info(f"[{i+1}/{num_queries}] {query_text[:80]}")

        # ── Baseline: clean KB ──
        baseline_resp = rag.query(query_text)
        baseline_result = QueryResult.from_rag_response(baseline_resp)
        baseline_results.append(baseline_result)

        # ── Attack injection ──
        clean_docs = baseline_resp.retrieved_docs
        adv_docs = attack.generate_adversarial_docs(query_text, clean_docs)
        n_before = len(retriever.documents)
        retriever.add_documents(adv_docs)

        try:
            attacked_resp = rag.query(query_text)
            attacked_result = QueryResult.from_rag_response(
                attacked_resp, baseline_latency=baseline_resp.total_latency_s
            )
            attacked_results.append(attacked_result)

            logger.info(
                f"  baseline: {baseline_result.refusal_type.value} "
                f"| attacked: {attacked_result.refusal_type.value} "
                f"| adv_retrieved: {attacked_result.retrieved_adversarial_count}/{len(adv_docs)}"
            )
        finally:
            # Remove adversarial docs for clean next-query state
            _rollback_index(retriever, n_before)

    loaded.unload()
    return {"baseline": baseline_results, "attacked": attacked_results}


def _rollback_index(retriever: HNSWRetriever, n_before: int) -> None:
    """Remove the last (len(current) - n_before) docs and rebuild index.

    FAISS HNSW doesn't support deletion, so we rebuild. For the prototype this
    is acceptable; for Thesis B, switch to marking docs as removed or use a
    deletion-friendly index.
    """
    if len(retriever.documents) == n_before:
        return
    kept_docs = retriever.documents[:n_before]
    # Rebuild index
    retriever.build_index(kept_docs)


def main():
    parser = argparse.ArgumentParser(description="Run a DoS attack experiment")
    parser.add_argument("--category", required=True, help="Attack category (A1, C1, ...)")
    parser.add_argument("--model-pair", required=True, help="Matched model pair name")
    parser.add_argument("--num-queries", type=int, default=None, help="Override query count")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--side", choices=["both", "base", "aligned"], default="both")
    args = parser.parse_args()

    env = detect_environment()
    logger.info(f"Environment: {env}")

    # Load configs
    pair = load_model_pair(args.model_pair)
    attack_config = load_attack(args.category)
    if args.num_queries is not None:
        attack_config.num_queries = args.num_queries

    logger.info(f"Model pair: {pair.pair_name}")
    logger.info(f"Attack: {args.category} ({attack_config.name})")
    logger.info(f"Queries: {attack_config.num_queries}, top_k: {args.top_k}")

    # Load data
    queries, kb_docs = load_dataset()
    logger.info(f"Loaded {len(queries)} queries, {len(kb_docs)} KB docs")

    # Build retriever (shared across both models)
    retriever = HNSWRetriever(embedder_id=args.embedder)
    retriever.build_index(kb_docs)

    # Build attack (needs embedder for C1)
    if args.category.startswith("C"):
        attack = build_attack(attack_config, embedder_id=args.embedder)
    else:
        attack = build_attack(attack_config)

    # Run on each side of the pair 
    all_results = {}
    metrics_reports = {}

    sides_to_run = []
    if args.side in ("both", "base"):
        sides_to_run.append(("base", pair.base))
    if args.side in ("both", "aligned"):
        sides_to_run.append(("aligned", pair.aligned))

    for side_name, model_cfg in sides_to_run:
        logger.info(f"══ Running {side_name.upper()} side: {model_cfg.name} ══")
        start = time.perf_counter()

        results = run_model_side(
            model_cfg,
            retriever=retriever,
            queries=queries,
            attack=attack,
            top_k=args.top_k,
            num_queries=attack_config.num_queries,
        )

        elapsed = time.perf_counter() - start
        logger.info(f"{side_name} side completed in {elapsed:.1f}s")

        all_results[side_name] = {
            "model_name": model_cfg.name,
            "hf_id": model_cfg.hf_id,
            "alignment_level": model_cfg.alignment_level,
            "baseline": [r.to_dict() for r in results["baseline"]],
            "attacked": [r.to_dict() for r in results["attacked"]],
        }

        report = compute_metrics(
            attacked=results["attacked"],
            baseline=results["baseline"],
            model_name=model_cfg.name,
            attack_category=args.category,
        )
        metrics_reports[side_name] = report.to_dict()

        # Print summary
        logger.info(
            f"  ASR={report.asr:.3f} | GDS={report.gds:.3f} | "
            f"LIR_mean={report.lir_mean:.2f} | TOR_mean={report.tor_mean:.2f} | "
            f"CDR={report.cdr:.3f}"
        )

    #  Save results 
    out_dir = RESULTS_DIR / f"{args.model_pair}_{args.category}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_reports, f, indent=2)

    logger.info(f"Saved results to {out_dir}/")
    logger.info(f"  raw_results.json — per-query outcomes")
    logger.info(f"  metrics.json — aggregated metrics per model")


if __name__ == "__main__":
    main()
