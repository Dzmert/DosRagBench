"""BLOCKER optimises on an independent embedding objective, not the severity
classifier it is scored with (no circularity), and mutates from a neutral vocab."""

from __future__ import annotations

import pytest

from dosragbench.attacks.blocker import _NEUTRAL_VOCAB, BlockerAttack, BlockerConfig
from dosragbench.attacks.blocker_construct import TARGET_RESPONSES


def _cfg(**kw):
    base = dict(n_tokens=4, batch_size=2, max_iters=3, stall_limit=2,
                seed=0, target="insufficient", device="cpu")
    base.update(kw)
    return BlockerConfig(**base)


class TestBlockerObjective:
    def test_requires_response_fn(self):
        attack = BlockerAttack(_cfg())
        with pytest.raises(RuntimeError):
            attack.generate_adversarial_docs("q", [])

    def test_unknown_target_rejected(self):
        with pytest.raises(ValueError):
            BlockerAttack(_cfg(target="bogus"))

    def test_response_fn_returns_text_and_blocker_built(self):
        attack = BlockerAttack(_cfg())
        calls = {"n": 0}

        def respond(query, adv_docs):
            calls["n"] += 1
            # Immediately emit the target refusal -> early stop before any batch.
            return TARGET_RESPONSES["insufficient"]

        attack.response_fn = respond
        docs = attack.generate_adversarial_docs("myquery", [])
        assert docs[0].text.startswith("myquery ")
        assert docs[0].is_adversarial
        # Target phrase present in the initial probe -> exactly one response call.
        assert calls["n"] == 1

    def test_optimises_toward_target_without_reading_severity(self):
        # response_fn never returns a denial phrase, so the search runs the full
        # (tiny) budget. It is scored purely by embedding similarity; the function
        # is never handed a severity value.
        attack = BlockerAttack(_cfg())
        seen_args = []

        def respond(query, adv_docs):
            seen_args.append((query, adv_docs))
            return "Paris is the capital of France."

        attack.response_fn = respond
        docs = attack.generate_adversarial_docs("q", [])
        assert docs and docs[0].text.startswith("q ")
        # response_fn only ever receives (query, docs) — never a severity int.
        assert all(isinstance(a[0], str) for a in seen_args)

    def test_vocab_is_neutral_not_refusal_biased(self):
        # The mutation pool must not be stacked with refusal words (which would
        # hand the search its target). Spot-check a few obvious ones are absent.
        for w in ("cannot", "refuse", "unable", "unknown", "restricted"):
            assert w not in _NEUTRAL_VOCAB
