"""Gold-passage scoring, including the multi-hop case.

`prepare_data.py` originally kept one gold passage per query. HotpotQA labels two
and needs both, so every HotpotQA query was indexed missing half its evidence and
its refusals were largely correct rather than adversarial -- see
docs/findings_summary.md §0.1. These tests pin the semantics that fix it: a
sequence of ids means ALL are required, and partial retrieval is not answerable.
"""

from __future__ import annotations

from dosragbench.pipeline.retriever import Document, score_gold


def docs(*ids: str) -> list[Document]:
    return [Document(doc_id=i, text=f"text of {i}") for i in ids]


def test_no_gold_requested():
    assert score_gold(docs("a", "b"), None) == (False, -1, [])


def test_single_gold_present():
    present, rank, ranks = score_gold(docs("a", "b", "c"), "b")
    assert (present, rank, ranks) == (True, 1, [1])


def test_single_gold_absent():
    present, rank, ranks = score_gold(docs("a", "b"), "zzz")
    assert present is False
    assert rank == -1
    assert ranks == [-1]


def test_single_gold_as_one_element_sequence_matches_string():
    assert score_gold(docs("a", "b"), "b") == score_gold(docs("a", "b"), ["b"])


def test_multi_hop_both_present_binds_on_worst_rank():
    # Answerable only once the SECOND hop arrives, so the rank is 3, not 0.
    present, rank, ranks = score_gold(docs("h1", "x", "y", "h2"), ["h1", "h2"])
    assert present is True
    assert rank == 3
    assert ranks == [0, 3]


def test_multi_hop_partial_is_not_answerable():
    # The single-gold bug: this used to count as gold-present.
    present, rank, ranks = score_gold(docs("h1", "x", "y"), ["h1", "h2"])
    assert present is False
    assert rank == -1
    assert ranks == [0, -1]


def test_multi_hop_neither_present():
    present, rank, ranks = score_gold(docs("x", "y"), ["h1", "h2"])
    assert present is False
    assert ranks == [-1, -1]


def test_rank_order_follows_the_requested_ids_not_the_results():
    _, _, ranks = score_gold(docs("h2", "h1"), ["h1", "h2"])
    assert ranks == [1, 0]


def test_empty_sequence_requests_nothing():
    assert score_gold(docs("a"), []) == (False, -1, [])
