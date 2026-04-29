"""Retriever component of the RAG pipeline.

Uses FAISS with HNSW indexing to mirror production vector databases
(Pinecone, Weaviate, Milvus, Qdrant all use HNSW).

Instrumented for retrieval latency measurement, which is the core
metric for Category C (algorithmic complexity) attacks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


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


@dataclass
class RetrievalResult:
    """Output of a retrieval, with timing info."""

    query: str
    documents: list[Document]
    scores: list[float]
    latency_s: float
    num_nodes_visited: int = 0  # HNSW graph traversal depth


class HNSWRetriever:
    """HNSW-based dense retriever using FAISS.

    HNSW parameters are set to production defaults (M=16, efConstruction=200).
    The efSearch parameter controls query-time traversal width — higher means
    more accurate but slower. Default efSearch=50.
    """

    def __init__(
        self,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        self.embedder_id = embedder_id
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        logger.info(f"Loading embedder: {embedder_id}")
        self.embedder = SentenceTransformer(embedder_id)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.index: Optional[faiss.Index] = None
        self.documents: list[Document] = []

    def build_index(self, documents: list[Document]) -> None:
        """Build HNSW index over the document collection."""
        logger.info(f"Building HNSW index ({len(documents)} docs, M={self.m})")
        self.documents = documents

        # Embed all documents
        texts = [doc.text for doc in documents]
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Build HNSW index (inner-product on normalized vectors = cosine)
        self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self.index.add(embeddings)

        logger.info(f"Index built: {self.index.ntotal} vectors, dim={self.dim}")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k documents with latency instrumentation.

        Returns the documents, their similarity scores, and the retrieval latency.
        This is the critical measurement point for Category C attacks.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Embed query
        query_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        # Time only the HNSW search itself (not embedding)
        start = time.perf_counter()
        scores, indices = self.index.search(query_emb, top_k)
        latency_s = time.perf_counter() - start

        documents = [self.documents[idx] for idx in indices[0] if idx >= 0]
        score_list = [float(s) for s in scores[0] if s > -1e9]

        return RetrievalResult(
            query=query,
            documents=documents,
            scores=score_list,
            latency_s=latency_s,
        )

    def add_documents(self, new_documents: list[Document]) -> None:
        """Add documents to the existing index (used for attack injection)."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        texts = [doc.text for doc in new_documents]
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.index.add(embeddings)
        self.documents.extend(new_documents)
