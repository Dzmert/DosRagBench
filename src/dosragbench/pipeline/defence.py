"""Retrieval-side defences, applied between candidate retrieval and top-k truncation.

Each defence maps a ranked candidate pool to a (possibly shorter, possibly
reordered) candidate pool; the retriever then keeps the first top_k. Defences
compose left to right, so `cluster,rerank` filters clusters first and reranks
what survives. They run on the clean baseline as well as the attacked query, so
the utility cost of a defence is measured on the same run as its benefit.

The pool is the top `pool_k` of the merged clean + adversarial candidates — what
a production system retrieving top-M before reranking would see. Adversarial
documents are not given any privileged treatment: if an attack fills the whole
pool, the defence only ever sees the attack, and that is what gets measured.

Two defences, each aimed at a mechanism the benchmark identified (§7 of
docs/findings_summary.md):

  rerank   A cross-encoder rescores (query, passage) pairs and reorders the pool.
           Targets the rank-0 result (§3.7): gold as the top hit roughly halves
           the alignment gap, so improving rank-1 precision should help most.

  cluster  Drops any candidate with at least `min_neighbours` near-duplicates
           (cosine > `tau`) in the pool. Targets the clustering mechanism (§3.9):
           C1 evicts gold 41x more than random injection at equal budget because
           its documents are packed together in embedding space. Natural passages
           rarely have several near-duplicates, so the clean cost should be low —
           but that is an empirical claim, measured by the clean run.
           Known gap: a cluster's edge member with fewer than `min_neighbours`
           neighbours above `tau` survives (seen in testing: one of six
           templated blockers was kept). Raising `min_neighbours` widens the gap;
           lowering `tau` raises the clean cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    score: float
    doc: "Document"  # noqa: F821  (dosragbench.pipeline.retriever.Document)
    embedding: Optional[np.ndarray] = None  # unit-normalised; set when a defence needs it


class RetrievalDefence:
    name = "none"
    needs_embeddings = False

    def apply(self, query: str, pool: list[Candidate]) -> list[Candidate]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


class CrossEncoderRerank(RetrievalDefence):
    name = "rerank"

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cpu",
                 scorer: Optional[Callable[[list[tuple[str, str]]], list[float]]] = None):
        self.model_id = model_id
        if scorer is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker: {model_id} (device={device})")
            model = CrossEncoder(model_id, device=device)
            scorer = lambda pairs: model.predict(pairs, batch_size=64).tolist()  # noqa: E731
        self._score = scorer

    def apply(self, query: str, pool: list[Candidate]) -> list[Candidate]:
        if not pool:
            return pool
        scores = self._score([(query, c.doc.text) for c in pool])
        rescored = [replace(c, score=float(s)) for c, s in zip(pool, scores)]
        return sorted(rescored, key=lambda c: c.score, reverse=True)

    def describe(self) -> dict:
        return {"name": self.name, "model": self.model_id}


class ClusterFilter(RetrievalDefence):
    name = "cluster"
    needs_embeddings = True

    def __init__(self, tau: float = 0.9, min_neighbours: int = 2):
        self.tau = tau
        self.min_neighbours = min_neighbours
        self.removed_total = 0  # running count, for the run log

    def apply(self, query: str, pool: list[Candidate]) -> list[Candidate]:
        if len(pool) <= self.min_neighbours:
            return pool
        emb = np.stack([c.embedding for c in pool])
        sims = emb @ emb.T
        neighbours = (sims > self.tau).sum(axis=1) - 1  # minus self
        kept = [c for c, n in zip(pool, neighbours) if n < self.min_neighbours]
        self.removed_total += len(pool) - len(kept)
        return kept

    def describe(self) -> dict:
        return {"name": self.name, "tau": self.tau, "min_neighbours": self.min_neighbours}


def build_defences(spec: str, device: str = "cpu", tau: float = 0.9,
                   min_neighbours: int = 2,
                   rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                   ) -> list[RetrievalDefence]:
    """Parse a comma-separated spec ("rerank", "cluster", "cluster,rerank")."""
    defences: list[RetrievalDefence] = []
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        if name == "rerank":
            defences.append(CrossEncoderRerank(rerank_model, device=device))
        elif name == "cluster":
            defences.append(ClusterFilter(tau=tau, min_neighbours=min_neighbours))
        else:
            raise ValueError(f"unknown defence {name!r}; expected 'rerank' or 'cluster'")
    return defences
