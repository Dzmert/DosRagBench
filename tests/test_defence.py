"""Tests for the retrieval defences (pipeline/defence.py) and their retriever hook."""

from __future__ import annotations

import numpy as np
import pytest

from dosragbench.pipeline.defence import (
    Candidate,
    ClusterFilter,
    CrossEncoderRerank,
    build_defences,
)
from dosragbench.pipeline.retriever import Document, HNSWRetriever


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _cand(doc_id, emb, score=0.0, adv=False):
    return Candidate(score=score, doc=Document(doc_id, doc_id, is_adversarial=adv),
                     embedding=_unit(emb))


class TestClusterFilter:
    def test_drops_dense_cluster_keeps_singletons(self):
        pool = [_cand(f"adv{i}", [1, 0.01 * i, 0], adv=True) for i in range(4)]
        pool += [_cand("clean_a", [0, 1, 0]), _cand("clean_b", [0, 0, 1])]
        kept = ClusterFilter(tau=0.9, min_neighbours=2).apply("q", pool)
        assert [c.doc.doc_id for c in kept] == ["clean_a", "clean_b"]

    def test_a_single_near_duplicate_pair_survives(self):
        # One neighbour is below min_neighbours=2: natural duplicates are tolerated.
        pool = [_cand("a", [1, 0, 0]), _cand("a_dup", [1, 0.01, 0]), _cand("b", [0, 1, 0])]
        kept = ClusterFilter(tau=0.9, min_neighbours=2).apply("q", pool)
        assert len(kept) == 3

    def test_preserves_order_and_counts_removals(self):
        pool = [_cand("x", [0, 1, 0], score=0.9)]
        pool += [_cand(f"c{i}", [1, 0.01 * i, 0], score=0.5) for i in range(3)]
        pool += [_cand("y", [0, 0, 1], score=0.1)]
        f = ClusterFilter(tau=0.9, min_neighbours=2)
        assert [c.doc.doc_id for c in f.apply("q", pool)] == ["x", "y"]
        assert f.removed_total == 3


class TestRerank:
    def test_reorders_by_scorer(self):
        scorer = lambda pairs: [len(text) for _, text in pairs]  # noqa: E731
        rr = CrossEncoderRerank(scorer=scorer)
        pool = [Candidate(0.9, Document("s", "short")), Candidate(0.1, Document("l", "much longer"))]
        assert [c.doc.doc_id for c in rr.apply("q", pool)] == ["l", "s"]

    def test_empty_pool(self):
        assert CrossEncoderRerank(scorer=lambda p: []).apply("q", []) == []


def test_build_defences_rejects_unknown():
    with pytest.raises(ValueError):
        build_defences("rerank,bogus")


class TestRetrieverHook:
    """Small real index: an injected near-duplicate cluster is filtered out."""

    CLEAN = [
        "Mount Vesuvius erupted in 79 AD and buried the Roman city of Pompeii.",
        "Lava from Kilauea has flowed into the sea several times since 1983.",
        "Miles Davis recorded Kind of Blue in 1959 with John Coltrane on saxophone.",
        "Individual tax returns in the United States are normally due on April 15.",
        "Brazil has won the FIFA World Cup five times, more than any other nation.",
        "Chlorophyll absorbs red and blue light and reflects green wavelengths.",
        "Augustus became the first Roman emperor after the defeat of Mark Antony.",
        "Lithium-ion cells dominate the battery packs of modern electric vehicles.",
        "The Sicilian Defence is the most popular response to 1.e4 in chess.",
        "The Great Barrier Reef is the largest coral reef system on Earth.",
        "Insulin is produced by beta cells in the pancreas.",
        "The Treaty of Versailles was signed in June 1919.",
    ]  # pairwise cosine <= 0.28: no natural near-duplicates

    @pytest.fixture(scope="class")
    def retriever(self):
        r = HNSWRetriever(device="cpu")
        r.build_index([Document(f"d{i}", t) for i, t in enumerate(self.CLEAN)])
        return r

    def _attack(self, r):
        # Pairwise cosine >= 0.97: every member has 5 near-duplicates.
        r.set_adversarial([
            Document(f"adv{i}", f"when did the volcano erupt? archived record {i}",
                     is_adversarial=True) for i in range(6)
        ])

    def test_undefended_path_unchanged(self, retriever):
        retriever.defences = []
        self._attack(retriever)
        res = retriever.retrieve("when did the volcano erupt", top_k=5)
        retriever.clear_adversarial()
        assert res.adversarial_in_topk == 5  # the cluster owns top-k without a defence

    def test_cluster_defence_removes_injected_cluster(self, retriever):
        retriever.defences = [ClusterFilter(tau=0.9, min_neighbours=2)]
        retriever.defence_pool_k = 20
        self._attack(retriever)
        res = retriever.retrieve("when did the volcano erupt", top_k=5)
        retriever.clear_adversarial()
        retriever.defences = []
        assert res.adversarial_in_topk == 0
        assert len(res.documents) == 5
        assert {d.doc_id for d in res.documents[:2]} == {"d0", "d1"}  # the volcano passages

    def test_clean_retrieval_untouched_by_cluster_defence(self, retriever):
        retriever.defences = []
        plain = [d.doc_id for d in retriever.retrieve("roman emperor", top_k=5).documents]
        retriever.defences = [ClusterFilter(tau=0.9, min_neighbours=2)]
        defended = [d.doc_id for d in retriever.retrieve("roman emperor", top_k=5).documents]
        retriever.defences = []
        assert defended == plain
