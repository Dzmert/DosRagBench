"""Answer-string correctness: token-boundary gold matching and the derived
accuracy / correctness-drop metrics."""

from __future__ import annotations

from dataclasses import dataclass

from dosragbench.metrics import QueryResult, answer_contains_gold, compute_metrics
from dosragbench.metrics.answer import answer_matches_any, gold_token_recall


@dataclass
class _Resp:
    query: str
    answer: str
    retrieval_latency_s: float = 0.001
    generation_latency_s: float = 0.01
    total_latency_s: float = 0.011
    output_tokens: int = 10
    input_tokens: int = 50
    retrieved_adversarial_count: int = 0
    gold_in_topk: bool = True
    gold_rank: int = 0


def _qr(answer, gold=""):
    return QueryResult.from_rag_response(_Resp(query="q", answer=answer), gold_answer=gold)


class TestTokenBoundary:
    def test_short_numeric_not_substring(self):
        assert not answer_contains_gold("It happened in 1945.", "4")
        assert answer_contains_gold("The answer is 4 apples.", "4")

    def test_short_word_not_inside_token(self):
        assert not answer_contains_gold("They started early.", "art")
        assert answer_contains_gold("It is modern art today.", "art")

    def test_multiword_subsequence(self):
        assert answer_contains_gold("It was Leonardo da Vinci.", "Leonardo da Vinci")
        assert not answer_contains_gold("Leonardo painted; da Vinci coded.", "Leonardo da Vinci")

    def test_empty_gold_is_unknown(self):
        assert not answer_contains_gold("anything", "")


class TestAccuracyMetrics:
    def test_accuracy_and_drop(self):
        gold = "da vinci"
        baseline = [_qr("Da Vinci.", gold), _qr("Da Vinci.", gold)]
        attacked = [_qr("Da Vinci.", gold), _qr("I don't know.", gold)]
        m = compute_metrics(attacked, baseline, "stub", "BLOCKER_INJECT")
        assert m.has_gold
        assert m.accuracy_baseline == 1.0
        assert m.accuracy_attacked == 0.5
        assert m.correctness_drop_rate == 0.5

    def test_goldless_corpus_is_unknown(self):
        baseline = [_qr("some answer"), _qr("another")]
        attacked = [_qr("x"), _qr("y")]
        m = compute_metrics(attacked, baseline, "stub", "C1")
        assert m.has_gold is False
        assert m.accuracy_baseline == 0.0
        assert m.correctness_drop_rate == 0.0


class TestMultiAlias:
    def test_any_alias_matches(self):
        golds = ["Roger Nichols", "Roger Nichols and Paul Williams"]
        assert answer_matches_any("It was written by Roger Nichols.", golds)
        assert not answer_matches_any("It was written by Carole King.", golds)

    def test_strict_misses_reworded_name_lenient_recovers(self):
        golds = ["Kirsten Simone Vangsness"]
        assert not answer_matches_any("Kirsten Vangsness", golds)
        assert answer_matches_any("Kirsten Vangsness", golds, lenient=True)

    def test_lenient_ignores_function_words(self):
        assert gold_token_recall("August 14, 1945", "14 and 15 August 1945") == 0.75

    def test_lenient_still_rejects_wrong_answer(self):
        assert not answer_matches_any("Abe Carver", ["Adam Caine"], lenient=True)
        assert not answer_matches_any("1977", ["October 20, 1977"], lenient=True)

    def test_no_aliases_is_never_correct(self):
        assert not answer_matches_any("anything", [])
        assert not answer_matches_any("anything", [], lenient=True)
