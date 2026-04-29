"""Unit tests for the refusal classifier.

Run with:
    pytest tests/test_refusal.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.metrics.refusal import (
    RefusalType,
    SeverityLevel,
    classify_refusal,
    classify_severity,
)


class TestExplicitSafetyRefusal:
    def test_cannot_answer(self):
        assert classify_refusal("I cannot answer this question.") == RefusalType.EXPLICIT_SAFETY

    def test_im_sorry_but(self):
        assert classify_refusal("I'm sorry, but I cannot help with that.") == RefusalType.EXPLICIT_SAFETY

    def test_not_able_to(self):
        assert classify_refusal("I'm not able to provide that information.") == RefusalType.EXPLICIT_SAFETY

    def test_must_decline(self):
        assert classify_refusal("I must decline to answer.") == RefusalType.EXPLICIT_SAFETY

    def test_against_my_guidelines(self):
        assert classify_refusal("That's against my guidelines.") == RefusalType.EXPLICIT_SAFETY


class TestEpistemicRefusal:
    def test_insufficient_context(self):
        result = classify_refusal("The provided context doesn't contain that information.")
        assert result == RefusalType.EPISTEMIC

    def test_cannot_determine(self):
        assert classify_refusal("I cannot determine the answer from the given text.") == RefusalType.EPISTEMIC

    def test_not_enough_information(self):
        assert classify_refusal("I don't have enough information to answer.") == RefusalType.EPISTEMIC


class TestHedgedNonAnswer:
    def test_many_hedges(self):
        text = "It depends on various factors. In some cases it might be true, but it's difficult to say without more context."
        assert classify_refusal(text) == RefusalType.HEDGED_NON_ANSWER


class TestGenerationFailure:
    def test_empty(self):
        assert classify_refusal("") == RefusalType.GENERATION_FAILURE

    def test_whitespace(self):
        assert classify_refusal("   ") == RefusalType.GENERATION_FAILURE

    def test_repetition_loop(self):
        text = "the answer is the answer is the answer is the answer is the answer is the answer is the answer is"
        assert classify_refusal(text) == RefusalType.GENERATION_FAILURE


class TestNoRefusal:
    def test_substantive_answer(self):
        text = "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose."
        assert classify_refusal(text) == RefusalType.NO_REFUSAL

    def test_factual_response(self):
        text = "The Eiffel Tower was completed in 1889 and stands 330 meters tall."
        assert classify_refusal(text) == RefusalType.NO_REFUSAL


class TestSeverity:
    def test_full_denial_from_safety_refusal(self):
        assert classify_severity("I cannot answer this.") == SeverityLevel.FULL_DENIAL

    def test_latency_degradation(self):
        # Substantive answer but slow
        result = classify_severity(
            "The answer is photosynthesis.",
            latency_inflation_ratio=5.0,
        )
        assert result == SeverityLevel.LATENCY_DEGRADATION

    def test_full_availability(self):
        assert classify_severity(
            "The Eiffel Tower was built in 1889.",
            latency_inflation_ratio=1.0,
        ) == SeverityLevel.FULL_AVAILABILITY

    def test_quality_degradation_from_epistemic(self):
        assert classify_severity(
            "The context doesn't contain that information.",
            latency_inflation_ratio=1.0,
        ) == SeverityLevel.QUALITY_DEGRADATION
