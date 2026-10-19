"""Answer-string correctness (distinct from gold *retrieval* scoring).

`retriever.score_gold` answers "was the gold passage retrieved?". This module
answers the separate question "did the generated answer actually contain the
gold short answer?", so a fluent-but-wrong or fluent-but-empty answer can be told
apart from a correct one. It is only meaningful on corpora whose queries carry a
short gold answer (NQ, HotpotQA); BEIR-style corpora leave `gold_answer` empty,
and then correctness is simply unknown (False), never falsely True.

Matching is token-level, not a raw character substring, so a short gold like "4"
or "art" does not spuriously match inside "1945" or "started".
"""

from __future__ import annotations

import re
import string

_ARTICLES = ("a", "an", "the")


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles, collapse spaces."""
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(?:%s)\b" % "|".join(_ARTICLES), " ", s)
    return " ".join(s.split())


def answer_contains_gold(answer: str, gold_answer: str) -> bool:
    """True if the normalized gold short answer appears in the normalized answer
    as a whole token subsequence. False when no gold answer is available."""
    if not gold_answer or not gold_answer.strip():
        return False
    gold_tokens = normalize_answer(gold_answer).split()
    if not gold_tokens:
        return False
    ans_tokens = normalize_answer(answer).split()
    if len(gold_tokens) > len(ans_tokens):
        return False
    first = gold_tokens[0]
    for i in range(len(ans_tokens) - len(gold_tokens) + 1):
        if ans_tokens[i] == first and ans_tokens[i:i + len(gold_tokens)] == gold_tokens:
            return True
    return False


_FUNCTION_WORDS = {"and", "or", "of", "in", "on", "at", "by", "to", "for", "with"}


def gold_token_recall(answer: str, gold_answer: str) -> float:
    """Fraction of the gold alias's content tokens that appear anywhere in the answer.

    The strict match misses correct answers worded differently from the alias:
    "Kirsten Vangsness" against gold "Kirsten Simone Vangsness", or "August 14,
    1945" against "14 and 15 August 1945". Recall over content tokens recovers
    those, at the cost of favouring long answers, which is why it is reported as
    a robustness check alongside the strict match rather than in place of it.
    """
    gold_tokens = [t for t in normalize_answer(gold_answer).split() if t not in _FUNCTION_WORDS]
    if not gold_tokens:
        return 0.0
    ans_tokens = set(normalize_answer(answer).split())
    return sum(t in ans_tokens for t in gold_tokens) / len(gold_tokens)


def answer_matches_any(answer: str, gold_answers: list[str], lenient: bool = False,
                       min_recall: float = 0.5) -> bool:
    """True if the answer matches any of the gold aliases (NQ-open lists several,
    e.g. "Roger Nichols" and "Roger Nichols and Paul Williams").

    Strict (default): an alias appears as a contiguous token sequence. Lenient:
    at least ``min_recall`` of an alias's content tokens appear in the answer.
    """
    if lenient:
        return any(gold_token_recall(answer, g) >= min_recall for g in gold_answers)
    return any(answer_contains_gold(answer, g) for g in gold_answers)
