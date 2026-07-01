"""Prepare a Natural Questions subset + knowledge base for DoSRAGBench.

Improvements over the prototype version:
  1. Scales to arbitrary query count and KB size via CLI flags.
  2. Records `gold_doc_id` in each query so C1 retrieval-displacement
     (recall@k drop) can be measured, not just adversarial pollution rate.
  3. Keeps the gold documents clearly separable from filler documents.

Usage (run on Katana where HuggingFace is reachable):
    python scripts/prepare_data.py --num-queries 200 --kb-size 100000

Outputs:
    data/queries.json         - [{query_id, query, gold_answer, gold_doc_id}]
    data/knowledge_base.json  - [{doc_id, text}]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.utils.config import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _make_doc_text(question: str, answer: str) -> str:
    """Format a Q/A pair as a knowledge-base document."""
    return (
        f"Topic: {question[:80]}\n\n"
        f"{answer} This information is relevant for queries about "
        f"{question.lower().rstrip('?')}."
    )


def load_nq_subset(num_queries: int, kb_size: int, seed: int = 42) -> dict:
    """Load a subset of Natural Questions with explicit gold-doc mapping."""
    try:
        from datasets import load_dataset

        logger.info("Loading Natural Questions (nq_open) from HuggingFace...")
        ds = load_dataset("google-research-datasets/nq_open", split="validation")

        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)

        if num_queries + kb_size > len(ds):
            logger.warning(
                f"Requested {num_queries + kb_size} rows but NQ validation has "
                f"{len(ds)}. Reducing KB size."
            )
            kb_size = max(0, len(ds) - num_queries)

        query_indices = indices[:num_queries]
        filler_indices = indices[num_queries : num_queries + kb_size]

        # Build filler documents first
        kb_docs = []
        for i, idx in enumerate(filler_indices):
            row = ds[idx]
            answer = row["answer"][0] if row["answer"] else ""
            kb_docs.append({
                "doc_id": f"kb_nq_{i:06d}",
                "text": _make_doc_text(row["question"], answer),
            })

        # Build gold documents and record the mapping in each query
        queries = []
        for i, idx in enumerate(query_indices):
            row = ds[idx]
            answer = row["answer"][0] if row["answer"] else ""
            gold_doc_id = f"kb_gold_{i:05d}"
            kb_docs.append({
                "doc_id": gold_doc_id,
                "text": _make_doc_text(row["question"], answer),
            })
            queries.append({
                "query_id": f"nq_{i:04d}",
                "query": row["question"],
                "gold_answer": answer,
                "gold_doc_id": gold_doc_id,   # <-- the key addition for C1
            })

        return {"queries": queries, "kb_docs": kb_docs, "source": "nq_open"}

    except Exception as exc:
        logger.warning(f"Could not load NQ ({exc}); using synthetic fallback.")
        return _synthetic_dataset(num_queries, kb_size, seed)


def _synthetic_dataset(num_queries: int, kb_size: int, seed: int) -> dict:
    """Fallback synthetic dataset (also carries gold_doc_id)."""
    rng = random.Random(seed)
    topics = [
        ("photosynthesis", "Photosynthesis converts sunlight into chemical energy."),
        ("eiffel tower", "The Eiffel Tower was completed in Paris in 1889."),
        ("mitochondria", "Mitochondria generate ATP as the cell's energy currency."),
        ("great wall", "The Great Wall of China spans over 13,000 miles."),
        ("einstein", "Albert Einstein developed the theory of relativity."),
        ("aspirin", "Aspirin reduces pain, fever, and inflammation."),
        ("everest", "Mount Everest is Earth's highest mountain at 8,849 metres."),
        ("dna", "DNA carries the genetic instructions for life."),
    ]
    kb_docs = []
    for i in range(kb_size):
        topic, ans = rng.choice(topics)
        kb_docs.append({"doc_id": f"kb_nq_{i:06d}", "text": _make_doc_text(f"about {topic}", ans)})

    queries = []
    for i in range(num_queries):
        topic, ans = rng.choice(topics)
        gold_doc_id = f"kb_gold_{i:05d}"
        kb_docs.append({"doc_id": gold_doc_id, "text": _make_doc_text(f"what is {topic}", ans)})
        queries.append({
            "query_id": f"syn_{i:04d}",
            "query": f"what is {topic}?",
            "gold_answer": ans,
            "gold_doc_id": gold_doc_id,
        })
    return {"queries": queries, "kb_docs": kb_docs, "source": "synthetic"}


def main():
    parser = argparse.ArgumentParser(description="Prepare DoSRAGBench dataset")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--kb-size", type=int, default=100000,
                        help="Number of filler docs (gold docs added on top)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_nq_subset(args.num_queries, args.kb_size, seed=args.seed)

    queries_path = args.output_dir / "queries.json"
    kb_path = args.output_dir / "knowledge_base.json"
    with open(queries_path, "w") as f:
        json.dump(data["queries"], f, indent=2)
    with open(kb_path, "w") as f:
        json.dump(data["kb_docs"], f, indent=2)

    logger.info(f"Wrote {len(data['queries'])} queries -> {queries_path}")
    logger.info(f"Wrote {len(data['kb_docs'])} KB docs -> {kb_path}")
    logger.info(f"  (filler + {len(data['queries'])} gold docs, source={data['source']})")


if __name__ == "__main__":
    main()