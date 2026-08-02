"""Inspect the clean (unattacked) refusals in a run's raw_results.json.

Motivation: Qwen 2.5 7B Instruct denies 54% of clean HotpotQA queries but only
17% of clean NQ queries, while Llama and Mistral barely move between datasets.
Attributable ASR conditions on the answerable subset, so a clean denial rate that
high makes the surviving queries a self-selected sample. Before any HotpotQA Qwen
number goes in the writeup we need to know *what* those clean refusals are.

Usage:
    python3 scripts/inspect_clean_refusals.py --run qwen-2.5-7b_A1
    python3 scripts/inspect_clean_refusals.py --run qwen-2.5-7b_A1 --results-dir results
"""

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(results_dir: str, run: str) -> dict:
    path = REPO / results_dir / run / "raw_results.json"
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    return json.load(open(path))


def counts(records: list) -> Counter:
    return Counter(r.get("refusal_type", "?") for r in records)


def pct(n: int, total: int) -> str:
    return f"{n:5d}  {100 * n / total:5.1f}%" if total else f"{n:5d}      -"


def report_side(side: str, data: dict, n_samples: int) -> None:
    baseline, attacked = data["baseline"], data["attacked"]
    print(f"\n{'=' * 68}\n{side.upper()}  ({data.get('model_name', '?')})\n{'=' * 68}")

    bc, ac = counts(baseline), counts(attacked)
    keys = sorted(set(bc) | set(ac))
    print(f"{'refusal_type':22}{'clean':>14}{'attacked':>14}")
    for k in keys:
        print(f"{k:22}{pct(bc[k], len(baseline)):>14}{pct(ac[k], len(attacked)):>14}")

    # Does a clean refusal track retrieval failure? If refusals concentrate where
    # gold is missing they are epistemic in substance, whatever the classifier
    # called them; if they are spread evenly the trigger is something else.
    print(f"\n{'clean refusal_type':22}{'gold in top-k':>15}{'gold missing':>14}")
    for k in sorted(bc):
        hit = sum(1 for r in baseline if r.get("refusal_type") == k and r.get("gold_in_topk"))
        print(f"{k:22}{hit:>15}{bc[k] - hit:>14}")

    for kind in ("explicit_safety", "epistemic"):
        samples = [r for r in baseline if r.get("refusal_type") == kind][:n_samples]
        if not samples:
            continue
        print(f"\n--- clean '{kind}' samples ({len(samples)} of {bc[kind]}) ---")
        for r in samples:
            print(f"\n  Q: {r.get('query', '')}")
            print(f"  gold_in_topk={r.get('gold_in_topk')} rank={r.get('gold_rank')} "
                  f"tokens={r.get('output_tokens')}")
            print(f"  A: {r.get('answer', '')[:400]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="qwen-2.5-7b_A1")
    ap.add_argument("--results-dir", default="results_hotpotqa")
    ap.add_argument("--side", default="aligned", choices=["aligned", "base", "both"])
    ap.add_argument("--n", type=int, default=6, help="samples per refusal type")
    args = ap.parse_args()

    data = load(args.results_dir, args.run)
    print(f"run: {args.results_dir}/{args.run}")
    sides = ["base", "aligned"] if args.side == "both" else [args.side]
    for side in sides:
        report_side(side, data[side], args.n)


if __name__ == "__main__":
    main()
