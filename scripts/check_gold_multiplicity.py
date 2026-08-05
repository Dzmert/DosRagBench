#!/usr/bin/env python3
"""How many gold passages does each query actually have, and how many did we keep?

`prepare_data.py` reduces qrels to ONE gold passage per query (`best_gold`, the
highest-scoring row). For NQ that is lossless -- NQ queries have a single
answer-bearing passage. For HotpotQA it is not: the questions are multi-hop and
need two passages, so keeping only the best-scoring one leaves the second hop
out of the corpus entirely.

That matters because it changes what a refusal means. If the evidence needed to
answer was never indexed, a model declining to answer is CORRECT, not
over-abstaining -- and the HotpotQA denial numbers are then partly measuring
missing evidence rather than alignment.

Run:
    python3 scripts/check_gold_multiplicity.py --dataset hotpotqa
    python3 scripts/check_gold_multiplicity.py --dataset nq

Reads only the qrels (a few MB), not the corpus. Safe on a login node.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

QRELS_REPO = {
    "nq": "BeIR/nq-qrels",
    "hotpotqa": "BeIR/hotpotqa-qrels",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(QRELS_REPO), default="hotpotqa")
    ap.add_argument(
        "--queries",
        type=pathlib.Path,
        default=None,
        help="Built queries.json to cross-check (default: data_<dataset>/queries.json)",
    )
    args = ap.parse_args()

    from datasets import load_dataset

    qrels = load_dataset(QRELS_REPO[args.dataset], split="test")

    per_query: dict[str, list[int]] = collections.defaultdict(list)
    for r in qrels:
        per_query[str(r["query-id"])].append(int(r["score"]))

    counts = collections.Counter(len(v) for v in per_query.values())
    total_gold = sum(len(v) for v in per_query.values())
    n_queries = len(per_query)

    print(f"dataset            : {args.dataset}")
    print(f"queries in qrels   : {n_queries}")
    print(f"gold rows in qrels : {total_gold}")
    print(f"mean gold/query    : {total_gold / max(n_queries, 1):.2f}")
    print("gold-per-query distribution:")
    for k in sorted(counts):
        print(f"  {k} gold passage(s): {counts[k]:6d} queries")

    kept = n_queries  # best_gold keeps exactly one per query
    print(f"\nprepare_data.py keeps 1 per query -> {kept} of {total_gold} gold rows")
    print(f"DISCARDED: {total_gold - kept} gold passages "
          f"({(total_gold - kept) / max(total_gold, 1):.1%} of the labelled evidence)")

    qpath = args.queries or pathlib.Path(f"data_{args.dataset}/queries.json")
    if args.dataset == "nq" and args.queries is None:
        qpath = pathlib.Path("data/queries.json")
    if qpath.exists():
        built = json.loads(qpath.read_text())
        built = built if isinstance(built, list) else built["queries"]
        has_multi = sum(1 for b in built if isinstance(b.get("gold_doc_id"), list))
        print(f"\nbuilt {qpath}: {len(built)} queries, "
              f"{has_multi} with a list-valued gold_doc_id "
              f"(0 confirms the single-gold reduction)")
    else:
        print(f"\n{qpath} not found -- skipped the built-corpus cross-check.")


if __name__ == "__main__":
    main()
