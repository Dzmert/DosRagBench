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
