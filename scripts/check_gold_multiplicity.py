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
    # Candidates for a third corpus. Check these BEFORE building one: a uniform
    # gold count above 1 means complementary evidence (like HotpotQA) and the
    # single-gold reduction would drop something the query needs. A ragged
    # distribution means redundant annotation (like NQ) and is harmless.
    "fiqa": "BeIR/fiqa-qrels",
    "trec-covid": "BeIR/trec-covid-qrels",
    "scifact": "BeIR/scifact-qrels",
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

    # A uniform count > 1 is the multi-hop signature: every query needs the same
    # number of passages, so the discarded ones are required evidence. A ragged
    # spread means annotators marked several passages that each answer the query
    # on their own, so dropping all but the best-scoring one costs nothing.
    print()
    blockers: list[str] = []

    if len(counts) == 1 and next(iter(counts)) > 1:
        blockers.append(
            f"COMPLEMENTARY evidence: a uniform {next(iter(counts))} gold passages for "
            "every query is the multi-hop signature, so the discarded ones are\n"
            "  required. Denial would be largely CORRECT refusal, not over-abstention."
        )

    # A ragged spread means redundant annotation, but that alone does not make a
    # corpus usable. TREC-COVID is ragged and still unusable: 50 topics with ~1,300
    # "gold" each is a document-ranking benchmark, where gold means topically
    # relevant rather than answer-bearing. Denial is not measurable against that.
    mean_gold = total_gold / max(n_queries, 1)
    if mean_gold > 20:
        blockers.append(
            f"RANKING BENCHMARK, not QA: {mean_gold:.0f} gold passages per query means "
            "'gold' marks\n  topical relevance, not an answer. A refusal cannot be "
            "scored as wrong against it."
        )
    if n_queries < 500:
        blockers.append(
            f"TOO FEW QUERIES: {n_queries} with qrels, against the 1,000 the existing "
            "runs use.\n  Cells would not be comparable to the NQ grid."
        )

    if blockers:
        print("VERDICT: NOT usable as-is --")
        for b in blockers:
            print(f"  - {b}")
    else:
        print("VERDICT: usable. Redundant annotation and enough queries; the kept")
        print("  passage still answers the query.")

    print("\nStructure is necessary but not sufficient -- also confirm the queries are")
    print("questions with answers in the corpus, not claims to verify or topics to rank.")

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
