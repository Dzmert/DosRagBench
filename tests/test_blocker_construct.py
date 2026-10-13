"""Tests for the construction-based blocker baselines (BLOCKER_INJECT / ORACLE)."""

from __future__ import annotations

import pytest

from dosragbench.attacks import build_attack
from dosragbench.attacks.blocker_construct import (
    INJECT_TEMPLATE,
    TARGET_RESPONSES,
    BlockerInjectAttack,
    BlockerOracleAttack,
)
from dosragbench.utils.config import AttackConfig


def _cfg(category, **params):
    return AttackConfig(category=category, name=category, num_queries=1,
                        num_adversarial_docs=1, seed=0, params=params)


class TestInject:
    def test_blocker_is_query_plus_injection(self):
        attack = BlockerInjectAttack(_cfg("BLOCKER_INJECT", target="insufficient"))
        docs = attack.generate_adversarial_docs("who discovered penicillin", [])
        assert len(docs) == 1
        assert docs[0].is_adversarial
        assert docs[0].text.startswith("who discovered penicillin")
        assert TARGET_RESPONSES["insufficient"] in docs[0].text

    def test_target_selectable(self):
        attack = BlockerInjectAttack(_cfg("BLOCKER_INJECT", target="safety"))
        assert TARGET_RESPONSES["safety"] in attack.generate_adversarial_docs("q", [])[0].text

    def test_unknown_target_rejected(self):
        with pytest.raises(ValueError):
            BlockerInjectAttack(_cfg("BLOCKER_INJECT", target="bogus"))

    def test_built_from_registry(self):
        attack = build_attack(_cfg("BLOCKER_INJECT"))
        assert isinstance(attack, BlockerInjectAttack)


class TestOracle:
    def test_requires_bound_oracle(self):
        attack = BlockerOracleAttack(_cfg("BLOCKER_ORACLE"))
        with pytest.raises(RuntimeError):
            attack.generate_adversarial_docs("q", [])

    def test_uses_oracle_output(self):
        attack = BlockerOracleAttack(_cfg("BLOCKER_ORACLE", oracle_words=30))
        attack.oracle_fn = lambda prompt: "A crafted passage inducing the target."
        docs = attack.generate_adversarial_docs("myquery", [])
        assert docs[0].text.startswith("myquery")
        assert "crafted passage" in docs[0].text

    def test_output_truncated_to_budget(self):
        attack = BlockerOracleAttack(_cfg("BLOCKER_ORACLE", oracle_words=5))
        attack.oracle_fn = lambda prompt: "word " * 100
        jamming = attack.generate_adversarial_docs("q", [])[0].text[len("q "):]
        assert len(jamming.split()) <= 5

    def test_reasoning_trace_stripped(self):
        attack = BlockerOracleAttack(_cfg("BLOCKER_ORACLE", oracle_words=50))
        attack.oracle_fn = lambda p: "<think>reason reason</think> The real passage."
        text = attack.generate_adversarial_docs("q", [])[0].text
        assert "reason reason" not in text
        assert "real passage" in text

    def test_fallback_on_oracle_error(self):
        attack = BlockerOracleAttack(_cfg("BLOCKER_ORACLE"))

        def boom(_p):
            raise RuntimeError("oom")

        attack.oracle_fn = boom
        docs = attack.generate_adversarial_docs("q", [])
        assert attack.fallbacks == 1
        assert INJECT_TEMPLATE.format(target=attack.target) in docs[0].text

    def test_fallback_on_empty_output(self):
        attack = BlockerOracleAttack(_cfg("BLOCKER_ORACLE"))
        attack.oracle_fn = lambda p: "   "
        docs = attack.generate_adversarial_docs("q", [])
        assert attack.fallbacks == 1
