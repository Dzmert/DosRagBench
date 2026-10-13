"""Tests for the jamming / blocker-document attack and the paired DoS metrics.

Run with:
    pytest tests/test_jamming.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.attacks.jamming import TARGET_RESPONSES, JammingAttack
from dosragbench.metrics import (
    QueryResult,
    answer_contains_gold,
    compute_avi,
    compute_metrics,
    response_answers_query,
)
from dosragbench.metrics.metrics import MetricsReport
from dosragbench.utils.config import AttackConfig


# ─── stub RAG response so QueryResult can be built without a model ───

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


def _cfg(method: str, **params) -> AttackConfig:
    return AttackConfig(
        category={"inject": "J1", "oracle": "J2", "opt": "J3"}[method],
        name=f"jam-{method}",
        num_queries=1,
        num_adversarial_docs=1,
        seed=0,
        params={"method": method, **params},
    )


# ─── blocker construction ───────────────────────────────────────────

class TestBlockerConstruction:
    def test_retrieval_part_is_query(self):
        attack = JammingAttack(_cfg("inject"))
        docs = attack.generate_adversarial_docs("who painted the mona lisa", [])
        assert len(docs) == 1
        assert docs[0].is_adversarial
        # The query is prepended as the retrieval sub-document.
        assert docs[0].text.startswith("who painted the mona lisa")

    def test_inject_contains_target_non_answer(self):
        attack = JammingAttack(_cfg("inject", target="insufficient"))
        docs = attack.generate_adversarial_docs("q", [])
        assert TARGET_RESPONSES["insufficient"] in docs[0].text

    def test_count_produces_multiple_copies(self):
        attack = JammingAttack(_cfg("inject"))
        docs = attack.generate_adversarial_docs("q", [], count=3)
        assert len(docs) == 3

    def test_unknown_method_rejected(self):
        import pytest

        bad = AttackConfig(category="J1", name="bad", params={"method": "bogus"})
        with pytest.raises(ValueError):
            JammingAttack(bad)


# ─── oracle (J2) and opt (J3) wiring ────────────────────────────────

class TestOracleAndOpt:
    def test_oracle_uses_callable(self):
        attack = JammingAttack(_cfg("oracle"))
        calls = []

        def fake_oracle(prompt: str) -> str:
            calls.append(prompt)
            return "Crafted passage that yields the target."

        docs = attack.generate_adversarial_docs("q", [], oracle_generate=fake_oracle)
        assert calls and "q" in calls[0]
        assert "Crafted passage" in docs[0].text

    def test_oracle_requires_callable(self):
        import pytest

        attack = JammingAttack(_cfg("oracle"))
        with pytest.raises(ValueError):
            attack.generate_adversarial_docs("q", [])

    def test_opt_terminates_and_requires_response_fn(self):
        import pytest

        attack = JammingAttack(_cfg("opt", opt_tokens=4, opt_iters=3, opt_batch=2))
        with pytest.raises(ValueError):
            attack.generate_adversarial_docs("q", [])

        # response_fn that immediately returns the target → early stop, valid doc.
        def obedient(_jamming_text: str) -> str:
            return TARGET_RESPONSES["insufficient"]

        docs = attack.generate_adversarial_docs("q", [], response_fn=obedient)
        assert len(docs) == 1
        assert docs[0].text.startswith("q ")


# ─── availability predicate: induced "insufficient info" is a non-answer ───

class TestAnswerPredicate:
    def test_insufficient_info_is_not_an_answer(self):
        assert response_answers_query(TARGET_RESPONSES["insufficient"]) is False

    def test_safety_refusal_is_not_an_answer(self):
        assert response_answers_query(TARGET_RESPONSES["safety"]) is False

    def test_substantive_answer_is_an_answer(self):
        assert response_answers_query("Leonardo da Vinci painted the Mona Lisa.") is True

    def test_gold_containment(self):
        assert answer_contains_gold("It was Leonardo da Vinci, the painter.", "Leonardo da Vinci")
        assert not answer_contains_gold("I don't know.", "Leonardo da Vinci")
        assert not answer_contains_gold("anything", "")  # gold unknown -> False


# ─── paired metrics ─────────────────────────────────────────────────

def _qr(answer, gold="", baseline_latency=None):
    return QueryResult.from_rag_response(
        _Resp(query="q", answer=answer), baseline_latency=baseline_latency, gold_answer=gold
    )


class TestPairedMetrics:
    def test_jam_success_is_baseline_corrected(self):
        # 4 queries. Clean: 3 answered, 1 already refusing.
        # Attacked: the refusing one stays refusing (not counted), 2 of the 3
        # answerable get jammed, 1 still answers.
        gold = "da vinci"
        baseline = [
            _qr("Da Vinci painted it.", gold),
            _qr("Da Vinci.", gold),
            _qr("The painter was Da Vinci.", gold),
            _qr(TARGET_RESPONSES["insufficient"], gold),
        ]
        attacked = [
            _qr(TARGET_RESPONSES["insufficient"], gold),  # jammed
            _qr(TARGET_RESPONSES["safety"], gold),        # jammed
            _qr("Still Da Vinci.", gold),                 # survived
            _qr(TARGET_RESPONSES["insufficient"], gold),  # was already non-answer
        ]
        m = compute_metrics(attacked, baseline, "stub", "J1")
        assert m.answerable_baseline == 3
        assert m.induced_denial_count == 2
        assert m.jam_success_rate == 2 / 3
        # Net denial: attacked non-answers (3) - baseline non-answers (1) = 2/4.
        assert m.net_denial_rate == 0.5

    def test_reverse_flip_detected(self):
        baseline = [_qr(TARGET_RESPONSES["insufficient"])]
        attacked = [_qr("Actually the answer is 42.")]
        m = compute_metrics(attacked, baseline, "stub", "J1")
        assert m.answerable_baseline == 0
        # Nothing was answerable clean, so jam success is undefined, not a measured 0.
        assert m.jam_success_rate is None
        assert m.reverse_flip_rate == 1.0

    def test_correctness_drop(self):
        gold = "da vinci"
        baseline = [_qr("Da Vinci.", gold), _qr("Da Vinci.", gold)]
        attacked = [_qr("Da Vinci.", gold), _qr("I don't know.", gold)]
        m = compute_metrics(attacked, baseline, "stub", "J1")
        assert m.has_gold
        assert m.accuracy_baseline == 1.0
        assert m.accuracy_attacked == 0.5
        assert m.correctness_drop_rate == 0.5


# ─── AVI zero-guard ─────────────────────────────────────────────────

class TestAVIZeroGuard:
    def _report(self, jam):
        r = MetricsReport(model_name="m", attack_category="J1", num_queries=10)
        r.jam_success_rate = jam
        return r

    def test_both_zero_is_undefined(self):
        avi = compute_avi(self._report(0.0), self._report(0.0))
        assert avi["avi_jam"] is None
        assert "undefined" in avi["interpretation"].lower()

    def test_real_paradox_defined(self):
        avi = compute_avi(self._report(0.6), self._report(0.2))
        assert avi["avi_jam"] == 3.0


# ─── runner wiring (_jamming_kwargs) without loading a real model ───

def _load_runner():
    import importlib.util

    path = REPO_ROOT / "scripts" / "run_attack.py"
    spec = importlib.util.spec_from_file_location("run_attack_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeModel:
    def generate(self, prompt):
        return {"text": "ORACLE:" + prompt[:10]}


class _FakeRag:
    def __init__(self):
        self.model = _FakeModel()
        self.last_query = None

    def query(self, q, gold_doc_id=None):
        self.last_query = q
        return _Resp(query=q, answer=TARGET_RESPONSES["insufficient"])


class _FakeRetriever:
    def __init__(self):
        self.adv = None

    def set_adversarial(self, docs):
        self.adv = docs


class TestRunnerWiring:
    def test_inject_needs_no_callables(self):
        mod = _load_runner()
        attack = JammingAttack(_cfg("inject"))
        kw = mod._jamming_kwargs(attack, _FakeRag(), _FakeRetriever(), "q", None)
        assert kw == {}

    def test_oracle_kwarg_is_model_generate(self):
        mod = _load_runner()
        attack = JammingAttack(_cfg("oracle"))
        kw = mod._jamming_kwargs(attack, _FakeRag(), _FakeRetriever(), "q", None)
        assert "oracle_generate" in kw
        assert kw["oracle_generate"]("hello world prompt").startswith("ORACLE:")

    def test_opt_response_fn_runs_rag_with_blocker(self):
        mod = _load_runner()
        attack = JammingAttack(_cfg("opt"))
        rag, retr = _FakeRag(), _FakeRetriever()
        kw = mod._jamming_kwargs(attack, rag, retr, "who", None)
        assert "response_fn" in kw
        out = kw["response_fn"]("filler tokens")
        # The probe blocker is set on the retriever and the RAG is queried.
        assert retr.adv and retr.adv[0].text.startswith("who filler tokens")
        assert rag.last_query == "who"
        assert out == TARGET_RESPONSES["insufficient"]


# ─── review follow-ups: boundary match, undefined AVI, truncation, caching ───

class TestGoldTokenBoundary:
    def test_short_numeric_gold_not_substring(self):
        assert not answer_contains_gold("It happened in 1945.", "4")
        assert answer_contains_gold("The answer is 4 apples.", "4")

    def test_short_word_gold_not_inside_token(self):
        assert not answer_contains_gold("They started early.", "art")
        assert answer_contains_gold("It is modern art today.", "art")

    def test_multiword_gold_subsequence(self):
        assert answer_contains_gold("It was Leonardo da Vinci, painter.", "Leonardo da Vinci")
        assert not answer_contains_gold("Leonardo painted; da Vinci coded.", "Leonardo da Vinci")


class TestUndefinedJamAVI:
    def test_none_jam_side_makes_avi_undefined(self):
        base = MetricsReport(model_name="b", attack_category="J1", num_queries=5)
        base.jam_success_rate = None  # base answered nothing clean
        aligned = MetricsReport(model_name="a", attack_category="J1", num_queries=5)
        aligned.jam_success_rate = 0.4
        avi = compute_avi(aligned, base)
        assert avi["avi_jam"] is None
        assert "undefined" in avi["interpretation"].lower()


class TestOracleHygiene:
    def test_oracle_output_truncated_to_word_budget(self):
        attack = JammingAttack(_cfg("oracle", oracle_words=5))
        long_text = "word " * 100
        docs = attack.generate_adversarial_docs("myquery", [], oracle_generate=lambda p: long_text)
        jamming = docs[0].text[len("myquery "):]
        assert len(jamming.split()) <= 5

    def test_reasoning_trace_stripped(self):
        attack = JammingAttack(_cfg("oracle", oracle_words=50))
        out = "<think>let me reason about this</think> The real crafted passage here."
        docs = attack.generate_adversarial_docs("q", [], oracle_generate=lambda p: out)
        assert "reason about this" not in docs[0].text
        assert "real crafted passage" in docs[0].text

    def test_fallback_counted(self):
        attack = JammingAttack(_cfg("oracle"))

        def boom(_p):
            raise RuntimeError("oom")

        attack.generate_adversarial_docs("q", [], oracle_generate=boom)
        assert attack._oracle_fallbacks == 1


class TestCrossSideCache:
    def test_same_blocker_reused_across_sides(self):
        attack = JammingAttack(_cfg("oracle"))
        calls = {"n": 0}

        def oracle(_p):
            calls["n"] += 1
            return f"crafted passage number {calls['n']}"

        first = attack.generate_adversarial_docs("q", [], oracle_generate=oracle)
        # Second side runs the same query: must reuse, not re-craft.
        second = attack.generate_adversarial_docs("q", [], oracle_generate=oracle)
        assert calls["n"] == 1
        assert first[0].text == second[0].text
