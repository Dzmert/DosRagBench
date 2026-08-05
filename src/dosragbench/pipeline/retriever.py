"""Retriever component of the RAG pipeline.

Uses FAISS with HNSW indexing to mirror production vector databases
(Pinecone, Weaviate, Milvus, Qdrant all use HNSW).

Efficient adversarial injection: the clean knowledge-base index is built
ONCE and never modified. Adversarial documents for a given query are placed
in a small ephemeral secondary index; retrieval searches both and merges by
score. This avoids the O(N) full-index rebuild that the naive prototype did
after every query, which is prohibitive at 100k+ document scale.

Two latency measurements are captured:
  - clean_search_latency: time to search the clean index only (baseline)
  - attacked_search_latency: time to search clean + adversarial and merge
These feed the Category C (algorithmic complexity) retrieval-LIR metric.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """'auto' -> cuda if available else cpu; 'cuda' with no GPU raises; else as-is."""
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not device or device == "auto":
        return "cuda" if cuda_ok else "cpu"

    if device.startswith("cuda") and not cuda_ok:
        raise RuntimeError(
            "device='cuda' was requested but no CUDA GPU is visible. On Katana you "
            "are most likely on a login node — GPUs only exist inside a PBS job that "
            "requests one. Start an interactive H200 session:\n"
            "    qsub -I -l select=1:ngpus=1:ncpus=8:mem=90gb:gpu_model=H200,walltime=04:00:00\n"
            "or submit scripts/run_c1_katana.pbs. For a quick login-node check use "
            "--device cpu (or --device auto, which falls back to CPU automatically)."
        )

    return device


@dataclass
class Document:
    """A document in the knowledge base."""

    doc_id: str
    text: str
    is_adversarial: bool = False
    attack_category: Optional[str] = None

    def __repr__(self):
        marker = "[ADV]" if self.is_adversarial else "     "
        return f"Document({marker} {self.doc_id}: {self.text[:60]}...)"


def score_gold(
    documents: list["Document"],
    gold_doc_id: Optional[Union[str, Sequence[str]]],
) -> tuple[bool, int, list[int]]:
    """Locate the required gold passage(s) in a ranked result list.

    Returns (all_present, binding_rank, per_passage_ranks). A sequence of ids means
    the query needs ALL of them -- HotpotQA is 2-hop, so retrieving one passage
    leaves the question unanswerable and must not count as gold-present. The
    binding rank is the worst of the required ranks, since the query only becomes
    answerable once the last one arrives; it is -1 if any is missing.

    Pulled out of `retrieve` so the semantics are unit-testable without an index.
    """
    if gold_doc_id is None:
        return False, -1, []
    required = [gold_doc_id] if isinstance(gold_doc_id, str) else list(gold_doc_id)
    if not required:
        return False, -1, []
    position = {d.doc_id: rank for rank, d in enumerate(documents)}
    ranks = [position.get(g, -1) for g in required]
    if all(r >= 0 for r in ranks):
        return True, max(ranks), ranks
    return False, -1, ranks


@dataclass
class RetrievalResult:
    """Output of a retrieval, with timing info."""

    query: str
    documents: list[Document]
    scores: list[float]
    latency_s: float
    adversarial_in_topk: int = 0
    # True only when EVERY required gold passage is in top-k. For single-gold
    # corpora (NQ) that is the old meaning unchanged. For complementary-evidence
    # corpora (HotpotQA, where a query needs both hops) partial retrieval is not
    # answerable, so it must not count as gold-present.
    gold_in_topk: bool = False
    # Rank at which the query becomes answerable: the WORST rank among the
    # required passages, since the last one to arrive is the binding constraint.
    # -1 if any required passage is absent. Equals the gold's rank for single-gold.
    gold_rank: int = -1
    gold_ranks: list[int] = field(default_factory=list)  # per required passage, in order


class HNSWRetriever:
    """HNSW dense retriever with efficient per-query adversarial injection.

    Usage:
        r = HNSWRetriever()
        r.build_index(clean_docs)            # once, expensive
        r.retrieve(query, top_k)             # clean baseline
        r.set_adversarial(adv_docs)          # cheap, per query
        r.retrieve(query, top_k)             # attacked
        r.clear_adversarial()                # cheap, per query
    """

    def __init__(
        self,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        device: str = "auto",
    ):
        self.embedder_id = embedder_id
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.device = _resolve_device(device)

        logger.info(f"Loading embedder: {embedder_id} (device={self.device})")
        self.embedder = SentenceTransformer(embedder_id, device=self.device)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # Clean index (built once)
        self.index: Optional[faiss.Index] = None
        self.documents: list[Document] = []

        # Ephemeral adversarial store (swapped per query)
        self._adv_docs: list[Document] = []
        self._adv_embeddings: Optional[np.ndarray] = None

    # ─── Index construction (expensive, done once) ───────────────────

    def build_index(self, documents: list[Document]) -> None:
        """Build the clean HNSW index. Call once."""
        logger.info(f"Building HNSW index ({len(documents)} docs, M={self.m})")
        self.documents = documents

        texts = [doc.text for doc in documents]
        embeddings = self._embed(texts, show_progress=True)

        self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self.index.add(embeddings)

        logger.info(f"Index built: {self.index.ntotal} vectors, dim={self.dim}")

    def save_index(self, path: str) -> None:
        """Persist the clean index + documents to disk for reuse across runs."""
        if self.index is None:
            raise RuntimeError("No index to save.")
        faiss.write_index(self.index, f"{path}.faiss")
        import json
        with open(f"{path}.docs.json", "w") as f:
            json.dump(
                [{"doc_id": d.doc_id, "text": d.text} for d in self.documents], f
            )
        logger.info(f"Saved index to {path}.faiss (+ .docs.json)")

    def load_index(self, path: str) -> None:
        """Load a previously-saved clean index, skipping re-embedding."""
        import json
        self.index = faiss.read_index(f"{path}.faiss")
        self.index.hnsw.efSearch = self.ef_search
        with open(f"{path}.docs.json") as f:
            raw = json.load(f)
        self.documents = [Document(doc_id=d["doc_id"], text=d["text"]) for d in raw]
        logger.info(f"Loaded index: {self.index.ntotal} vectors from {path}.faiss")

    # ─── Per-query adversarial injection (cheap) ─────────────────────

    def set_adversarial(self, adv_docs: list[Document]) -> None:
        """Register adversarial docs for the next retrieval(s). Cheap."""
        self._adv_docs = adv_docs
        if adv_docs:
            self._adv_embeddings = self._embed([d.text for d in adv_docs])
        else:
            self._adv_embeddings = None

    def clear_adversarial(self) -> None:
        """Remove adversarial docs. Cheap; no index rebuild."""
        self._adv_docs = []
        self._adv_embeddings = None

    # ─── Retrieval ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        gold_doc_id: Optional[Union[str, Sequence[str]]] = None,
    ) -> RetrievalResult:
        """Retrieve top-k from clean index merged with any adversarial docs.

        Latency measured covers the clean HNSW search plus (if present) the
        adversarial brute-force search and merge. When no adversarial docs are
        set, this is a pure clean-index search = the baseline measurement.

        `gold_doc_id` accepts a single id or a sequence of them. A sequence means
        the query needs ALL of those passages to be answerable (multi-hop), so
        `gold_in_topk` is True only when every one is retrieved. Passing a single
        id behaves exactly as before.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index().")

        query_emb = self._embed([query])

        start = time.perf_counter()

        # Search the clean index
        clean_scores, clean_idx = self.index.search(query_emb, top_k)
        candidates = []
        for score, idx in zip(clean_scores[0], clean_idx[0]):
            if idx < 0:
                continue
            candidates.append((float(score), self.documents[idx]))

        # Search adversarial docs (brute force; there are few)
        if self._adv_embeddings is not None and len(self._adv_docs) > 0:
            adv_scores = (self._adv_embeddings @ query_emb[0]).astype(float)
            for score, doc in zip(adv_scores, self._adv_docs):
                candidates.append((float(score), doc))

        # Merge: highest inner-product score wins, take top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:top_k]

        latency_s = time.perf_counter() - start

        documents = [doc for _, doc in top]
        scores = [s for s, _ in top]

        # Analytics for Category C displacement metrics
        adv_in_topk = sum(1 for d in documents if d.is_adversarial)
        gold_in_topk, gold_rank, gold_ranks = score_gold(documents, gold_doc_id)

        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores,
            latency_s=latency_s,
            adversarial_in_topk=adv_in_topk,
            gold_in_topk=gold_in_topk,
            gold_rank=gold_rank,
            gold_ranks=gold_ranks,
        )

    # ─── Helpers ─────────────────────────────────────────────────────

    def _embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)