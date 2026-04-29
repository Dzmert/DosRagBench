"""Prepare a sample Natural Questions subset + knowledge base for the prototype.

Usage:
    python scripts/prepare_data.py

Outputs:
    data/queries.json - list of {query_id, query, gold_answer} objects
    data/knowledge_base.json - list of {doc_id, text} objects for the KB
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Allow running from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.utils.config import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_nq_subset(num_queries: int, kb_size: int, seed: int = 42) -> dict:
    """Load a subset of Natural Questions for the prototype.

    Uses the 'natural_questions_open' variant for easier answer handling.
    Falls back to a synthetic toy dataset if the HF dataset can't be loaded.
    """
    try:
        from datasets import load_dataset

        logger.info("Loading Natural Questions (nq_open) from HuggingFace...")
        ds = load_dataset("nq_open", split="validation")

        # Sample queries
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        sample_indices = indices[:num_queries]

        queries = []
        for i, idx in enumerate(sample_indices):
            row = ds[idx]
            queries.append({
                "query_id": f"nq_{i:04d}",
                "query": row["question"],
                "gold_answer": row["answer"][0] if row["answer"] else "",
            })

        # Build a KB from the remaining queries' gold answers (as pseudo-documents)
        # This gives us a realistic mix of topically relevant content
        kb_docs = []
        kb_indices = indices[num_queries : num_queries + kb_size]
        for i, idx in enumerate(kb_indices):
            row = ds[idx]
            answer = row["answer"][0] if row["answer"] else ""
            question = row["question"]
            # Treat each Q/A pair as a mini document
            doc_text = (
                f"Topic: {question[:80]}\n\n"
                f"{answer} This information is relevant for queries about {question.lower().rstrip('?')}."
            )
            kb_docs.append({
                "doc_id": f"kb_nq_{i:05d}",
                "text": doc_text,
            })

        # Also add the gold answer docs for the query set (so baseline RAG works)
        for i, idx in enumerate(sample_indices):
            row = ds[idx]
            answer = row["answer"][0] if row["answer"] else ""
            question = row["question"]
            doc_text = (
                f"Topic: {question[:80]}\n\n"
                f"{answer} This information is relevant for queries about {question.lower().rstrip('?')}."
            )
            kb_docs.append({
                "doc_id": f"kb_gold_{i:04d}",
                "text": doc_text,
            })

        return {"queries": queries, "kb_docs": kb_docs, "source": "nq_open"}

    except Exception as exc:
        logger.warning(f"Could not load NQ dataset ({exc}); falling back to synthetic data.")
        return _synthetic_dataset(num_queries, kb_size, seed)


def _synthetic_dataset(num_queries: int, kb_size: int, seed: int) -> dict:
    """Fallback synthetic dataset for when HF is unavailable."""
    rng = random.Random(seed)

    topics = [
        ("photosynthesis", "Photosynthesis is the process plants use to convert sunlight into energy."),
        ("eiffel tower", "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889."),
        ("mitochondria", "Mitochondria are organelles that generate ATP as the cell's energy currency."),
        ("great wall", "The Great Wall of China stretches over 13,000 miles across northern China."),
        ("einstein", "Albert Einstein developed the theory of relativity and won the 1921 Nobel Prize in Physics."),
        ("aspirin", "Aspirin is a medication used to reduce pain, fever, or inflammation."),
        ("mount everest", "Mount Everest is Earth's highest mountain at 8,849 meters above sea level."),
        ("dna", "DNA is a molecule that carries the genetic instructions for life."),
        ("shakespeare", "William Shakespeare was an English playwright who wrote around 39 plays."),
        ("amazon river", "The Amazon River in South America is the largest river by discharge volume."),
        ("pluto", "Pluto is a dwarf planet in the Kuiper belt, reclassified from planet status in 2006."),
        ("penicillin", "Penicillin was discovered by Alexander Fleming in 1928 as the first antibiotic."),
        ("coffee", "Coffee is a brewed beverage prepared from roasted Coffea plant beans."),
        ("solar system", "The Solar System consists of the Sun and the objects that orbit it."),
        ("quantum", "Quantum mechanics describes physical properties at the scale of atoms and subatomic particles."),
    ]

    query_templates = [
        "What is {}?",
        "Can you explain {}?",
        "Tell me about {}.",
        "What are the key facts about {}?",
    ]

    queries = []
    for i in range(num_queries):
        topic, answer = rng.choice(topics)
        template = rng.choice(query_templates)
        queries.append({
            "query_id": f"syn_{i:04d}",
            "query": template.format(topic),
            "gold_answer": answer,
        })

    kb_docs = []
    for i in range(kb_size + num_queries):
        topic, answer = rng.choice(topics)
        filler_a = rng.choice(["Historical context suggests", "Research indicates", "Studies show"])
        filler_b = rng.choice(["This is well documented", "Experts confirm this view", "Evidence supports this"])
        kb_docs.append({
            "doc_id": f"syn_kb_{i:05d}",
            "text": f"{filler_a} that {answer} {filler_b}. Topic: {topic}.",
        })

    return {"queries": queries, "kb_docs": kb_docs, "source": "synthetic"}


def main():
    parser = argparse.ArgumentParser(description="Prepare prototype dataset")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of target queries")
    parser.add_argument("--kb-size", type=int, default=1000, help="Knowledge base size")
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

    logger.info(f"Wrote {len(data['queries'])} queries to {queries_path}")
    logger.info(f"Wrote {len(data['kb_docs'])} KB docs to {kb_path}")
    logger.info(f"Source: {data['source']}")


if __name__ == "__main__":
    main()
