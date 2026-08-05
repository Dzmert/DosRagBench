#!/usr/bin/env python3
"""Denial rate restricted to queries whose gold passage survived into top-k.

This answers the circularity objection, which is the strongest available attack
on the thesis:

    "You defined the vulnerability into existence. Context-faithfulness means
    'do not answer when the context does not support an answer'. You then
    degraded the context. A model declining to answer is behaving CORRECTLY --
    you have measured a working system, not a failure."

The objection is valid wherever the gold passage was evicted from top-k: there
the evidence really is gone and refusing really is right. It does NOT apply where
the gold passage is still in the retrieved context, because faithfulness would
answer from it. Splitting on `gold_in_topk` separates the two populations, and
only the gold-present half is evidence of over-abstention.

Labels are recomputed live from the answer text via `refusal.classify_refusal` --
the `refusal_type` field stored in raw_results.json is the OLD broken classifier's
output, frozen at run time, and must not be used.

Run:
    python3 scripts/gold_present_conditioning.py
    python3 scripts/gold_present_conditioning.py --results-root results_hotpotqa
    python3 scripts/gold_present_conditioning.py --per-run
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dosragbench.metrics.refusal import classify_refusal, is_denial  # noqa: E402


def denial(record: dict) -> bool:
    return is_denial(classify_refusal(record.get("answer", "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--per-run", action="store_true",
                    help="Also print a per-run breakdown, not just the pooled total.")
    args = ap.parse_args()

    pooled: dict[tuple[str, str, bool], list[int]] = collections.defaultdict(
        lambda: [0, 0]
    )
    per_run: dict[str, dict] = {}
    n_runs = 0

    for path in sorted(glob.glob(f"{args.results_root}/*/raw_results.json")):
        try:
            data = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if "base" not in data or "aligned" not in data:
            continue
        n_runs += 1
        run = os.path.basename(os.path.dirname(path))
        local: dict[tuple[str, str, bool], list[int]] = collections.defaultdict(
            lambda: [0, 0]
        )
        for side in ("base", "aligned"):
            for cond in ("baseline", "attacked"):
                for r in data[side].get(cond, []):
                    key = (side, cond, bool(r.get("gold_in_topk")))
                    for bucket in (pooled[key], local[key]):
                        bucket[1] += 1
                        if denial(r):
                            bucket[0] += 1
        per_run[run] = local

    def rate(side: str, cond: str, gold: bool, table=None) -> tuple[float, int]:
        d, t = (table or pooled)[(side, cond, gold)]
        return (d / t if t else float("nan")), t

    print(f"results root : {args.results_root}")
    print(f"runs pooled  : {n_runs}\n")

    for gold, label in ((True, "GOLD PRESENT in top-k"), (False, "GOLD EVICTED from top-k")):
        print(f"--- {label} ---")
        for side in ("base", "aligned"):
            for cond in ("baseline", "attacked"):
                r, n = rate(side, cond, gold)
                print(f"  {side:8s} {cond:9s}  denial {r:.3f}  (n={n})")
        ab, _ = rate("aligned", "baseline", gold)
        aa, _ = rate("aligned", "attacked", gold)
        bb, _ = rate("base", "baseline", gold)
        ba, _ = rate("base", "attacked", gold)
        print(f"  alignment effect, clean    : {ab - bb:+.3f}")
        print(f"  alignment effect, attacked : {aa - ba:+.3f}")
        print(f"  attack effect, base        : {ba - bb:+.3f}")
        print(f"  attack effect, aligned     : {aa - ab:+.3f}\n")

    if args.per_run:
        print("--- per run, GOLD PRESENT only, attacked ---")
        for run, local in sorted(per_run.items()):
            a_d, a_t = local[("aligned", "attacked", True)]
            b_d, b_t = local[("base", "attacked", True)]
            if not a_t or not b_t:
                continue
            print(f"  {run:44s} aligned {a_d/a_t:.3f}  base {b_d/b_t:.3f}  "
                  f"diff {a_d/a_t - b_d/b_t:+.3f}")

    print("\nCaveat: this pools all attacks and model pairs UNWEIGHTED, and is not")
    print("conditioned on clean-answerability the way the headline ASR is. It is a")
    print("robustness check on the circularity objection, not a replacement effect size.")


if __name__ == "__main__":
    main()
