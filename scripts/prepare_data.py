"""Prepare a query set + knowledge base for DoSRAGBench.

Corpus modes (--corpus):
  beir       - real BEIR NQ Wikipedia passages + qrels gold labels (default).
  synthetic  - templated docs derived from queries; smoke tests only.

Outputs (schema identical across modes):
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


# ─────────────────────────────── synthetic path ───────────────────────────────

def _make_doc_text(question: str, answer: str) -> str:
    """Format a Q/A pair as a knowledge-base document (synthetic mode only)."""
    return (
        f"Topic: {question[:80]}\n\n"
        f"{answer} This information is relevant for queries about "
        f"{question.lower().rstrip('?')}."
    )


def build_synthetic(num_queries: int, kb_size: int, seed: int = 42) -> dict:
    """Synthesize documents from NQ Q/A pairs (corpus derived from queries; smoke tests only)."""
    try:
        from datasets import load_dataset

        logger.info("Loading Natural Questions (nq_open) from HuggingFace...")
        ds = load_dataset("google-research-datasets/nq_open", split="train")

        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)

        if num_queries + kb_size > len(ds):
            logger.warning(
                f"Requested {num_queries + kb_size} rows but NQ train has "
                f"{len(ds)}. Reducing KB size."
            )
            kb_size = max(0, len(ds) - num_queries)

        query_indices = indices[:num_queries]
        filler_indices = indices[num_queries : num_queries + kb_size]

        kb_docs = []
        for i, idx in enumerate(filler_indices):
            row = ds[idx]
            answer = row["answer"][0] if row["answer"] else ""
            kb_docs.append({
                "doc_id": f"kb_nq_{i:06d}",
                "text": _make_doc_text(row["question"], answer),
            })

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
                "gold_doc_id": gold_doc_id,
            })

        return {"queries": queries, "kb_docs": kb_docs, "source": "synthetic:nq_open"}

    except Exception as exc:
        logger.warning(f"Could not load NQ ({exc}); using offline synthetic fallback.")
        return _synthetic_offline(num_queries, kb_size, seed)


def _synthetic_offline(num_queries: int, kb_size: int, seed: int) -> dict:
    """Fully offline synthetic dataset (no download); also carries gold_doc_id."""
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
    return {"queries": queries, "kb_docs": kb_docs, "source": "synthetic:offline"}


# ───────────────────────────────── BEIR path ──────────────────────────────────

# dataset name -> (corpus+queries repo, qrels repo)
BEIR_DATASETS = {
    "nq": ("BeIR/nq", "BeIR/nq-qrels"),
    "hotpotqa": ("BeIR/hotpotqa", "BeIR/hotpotqa-qrels"),
}

# Rough download sizes (parquet), logged before we fetch anything large.
BEIR_DOWNLOAD_HINT = {
    "nq": "~764 MB (corpus) + tiny qrels",
    "hotpotqa": "~1.5 GB (corpus) + tiny qrels",
}


def _passage_text(row: dict) -> str:
    """BEIR corpus passage -> 'Title. body' (standard BEIR usage)."""
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title}. {text}" if title else text


def build_beir(dataset: str, num_queries: int, kb_size: int, seed: int = 42) -> dict:
    """Build a BEIR dataset: KB = gold passages (from qrels) + `kb_size` random filler."""
    from datasets import load_dataset

    if dataset not in BEIR_DATASETS:
        raise ValueError(
            f"Unknown BEIR dataset '{dataset}'. Available: {', '.join(BEIR_DATASETS)}"
        )
    corpus_repo, qrels_repo = BEIR_DATASETS[dataset]

    logger.info(
        f"BEIR mode: dataset='{dataset}'. This will download {BEIR_DOWNLOAD_HINT[dataset]} "
        f"to your HuggingFace cache on first run."
    )

    # 1. qrels: query_id -> best gold corpus_id (highest relevance score)
    logger.info(f"Loading qrels: {qrels_repo} (split=test)")
    qrels_ds = load_dataset(qrels_repo, split="test")
    best_gold: dict[str, tuple[int, str]] = {}
    for r in qrels_ds:
        qid = str(r["query-id"] if "query-id" in r else r["query_id"])
        cid = str(r["corpus-id"] if "corpus-id" in r else r["corpus_id"])
        score = int(r.get("score", 1))
        if qid not in best_gold or score > best_gold[qid][0]:
            best_gold[qid] = (score, cid)

    # 2. queries: id -> text (only keep queries that have a gold label)
    logger.info(f"Loading queries: {corpus_repo} (config=queries)")
    queries_ds = load_dataset(corpus_repo, "queries", split="queries")
    q_text = {str(r["_id"]): r["text"] for r in queries_ds}

    eligible = [qid for qid in q_text if qid in best_gold]
    if not eligible:
        raise RuntimeError("No queries with qrels found; cannot build gold labels.")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected_qids = eligible[:num_queries]
    if len(selected_qids) < num_queries:
        logger.warning(
            f"Only {len(selected_qids)} queries have qrels; requested {num_queries}."
        )

    gold_ids = {best_gold[qid][1] for qid in selected_qids}

    # 3. corpus: resolve gold passages + sample filler passages
    logger.info(f"Loading corpus: {corpus_repo} (config=corpus)")
    corpus = load_dataset(corpus_repo, "corpus", split="corpus")
    n_corpus = len(corpus)
    logger.info(f"Corpus has {n_corpus} passages; building id->position map...")

    corpus_ids = corpus["_id"]  # fast Arrow column read
    id_to_pos = {str(cid): i for i, cid in enumerate(corpus_ids)}

    gold_positions = {id_to_pos[g] for g in gold_ids if g in id_to_pos}
    missing_gold = [g for g in gold_ids if g not in id_to_pos]
    if missing_gold:
        logger.warning(f"{len(missing_gold)} gold ids not found in corpus (dropped).")

    # Sample filler positions (excluding gold), targeting `kb_size` fillers.
    want_filler = min(kb_size, max(0, n_corpus - len(gold_positions)))
    # Oversample then filter out any gold collisions, then trim.
    pool_n = min(n_corpus, want_filler + len(gold_positions) + 1000)
    sampled = rng.sample(range(n_corpus), pool_n)
    filler_positions = [p for p in sampled if p not in gold_positions][:want_filler]

    needed = sorted(set(filler_positions) | gold_positions)
    logger.info(
        f"Materializing {len(needed)} passages "
        f"({len(gold_positions)} gold + {len(filler_positions)} filler)..."
    )
    subset = corpus.select(needed)

    kb_docs = []
    for row in subset:
        kb_docs.append({"doc_id": str(row["_id"]), "text": _passage_text(row)})

    # 4. queries with real gold_doc_id (drop any whose gold vanished)
    kb_id_set = {d["doc_id"] for d in kb_docs}
    queries = []
    for i, qid in enumerate(selected_qids):
        gold_doc_id = best_gold[qid][1]
        if gold_doc_id not in kb_id_set:
            continue
        queries.append({
            "query_id": qid,
            "query": q_text[qid],
            "gold_answer": "",  # BEIR queries carry no short answer; unused by C1
            "gold_doc_id": gold_doc_id,
        })

    return {"queries": queries, "kb_docs": kb_docs, "source": f"beir:{dataset}"}


# ─────────────────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare DoSRAGBench dataset")
    parser.add_argument("--corpus", choices=["beir", "synthetic"], default="beir",
                        help="beir = real Wikipedia passages + qrels (default); "
                             "synthetic = templated docs derived from queries (smoke tests)")
    parser.add_argument("--dataset", default="nq",
                        help="BEIR dataset name (nq, hotpotqa); ignored for --corpus synthetic")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--kb-size", type=int, default=500000,
                        help="Number of filler docs (gold docs added on top)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.corpus == "beir":
        data = build_beir(args.dataset, args.num_queries, args.kb_size, seed=args.seed)
    else:
        data = build_synthetic(args.num_queries, args.kb_size, seed=args.seed)

    queries_path = args.output_dir / "queries.json"
    kb_path = args.output_dir / "knowledge_base.json"
    with open(queries_path, "w") as f:
        json.dump(data["queries"], f, indent=2)
    with open(kb_path, "w") as f:
        json.dump(data["kb_docs"], f, indent=2)

    logger.info(f"Wrote {len(data['queries'])} queries -> {queries_path}")
    logger.info(f"Wrote {len(data['kb_docs'])} KB docs -> {kb_path}")
    logger.info(f"  (source={data['source']}, {len(data['queries'])} gold docs tracked)")


if __name__ == "__main__":
    main()
