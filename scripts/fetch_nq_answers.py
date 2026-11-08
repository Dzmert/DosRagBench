"""Attach NQ-open short answers to the BEIR NQ queries.

BEIR NQ ships queries and qrels but no short answers, so `prepare_data.py`
writes `gold_answer: ""` and the benchmark has had no answer-correctness ground
truth at all. BEIR NQ's queries are the Natural Questions dev set, and NQ-open
(Lee et al. 2019) is the subset of NQ whose questions have a short answer of at
most five tokens, with every annotated alias kept. So the two can be joined on
the question text.

The join is exact on a normalised question (lowercased, trailing "?" and
surrounding whitespace removed). No fuzzy matching: a near-miss join would
attach a different question's answer, which is worse than no answer. Queries
that NQ-open dropped — those whose only answers are long passages — stay
unmatched and are excluded from correctness scoring rather than scored wrong.

Writes `<data-dir>/nq_gold_answers.json`:  {query_id: [alias, ...]}.
It is a side file, deliberately not merged into queries.json, so the run inputs
on Katana and the KB signature are untouched.

Usage:
    python scripts/fetch_nq_answers.py                  # data/ (BEIR NQ)
    python scripts/fetch_nq_answers.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def normalise_question(q: str) -> str:
    return q.strip().lower().rstrip("?").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = ap.parse_args()

    from datasets import load_dataset

    index: dict[str, tuple[str, list[str]]] = {}
    # Validation first: BEIR NQ is drawn from NQ dev, which is NQ-open's validation
    # split. Train is a fallback for any question NQ-open placed there.
    for split in ("validation", "train"):
        for row in load_dataset("google-research-datasets/nq_open", split=split):
            index.setdefault(normalise_question(row["question"]), (split, list(row["answer"])))

    queries = json.loads((args.data_dir / "queries.json").read_text(encoding="utf-8"))
    gold: dict[str, list[str]] = {}
    by_split = {"validation": 0, "train": 0}
    for q in queries:
        hit = index.get(normalise_question(q["query"]))
        if hit:
            split, answers = hit
            gold[q["query_id"]] = answers
            by_split[split] += 1

    out = args.data_dir / "nq_gold_answers.json"
    out.write_text(json.dumps(gold, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Matched {len(gold)}/{len(queries)} queries to NQ-open "
          f"(validation {by_split['validation']}, train {by_split['train']}); wrote {out}")


if __name__ == "__main__":
    main()
