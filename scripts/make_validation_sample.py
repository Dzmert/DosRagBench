"""Sample responses for hand-labelling, to validate the refusal classifier.

Usage:
    python scripts/make_validation_sample.py
    python scripts/make_validation_sample.py --run llama-3.1-8b_D3 --n 100

Writes validation/sample.csv with an empty my_label column.
Fill my_label with: refusal | answer
Then run scripts/score_validation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "validation"

FULL_DENIAL = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="llama-3.1-8b_D3")
    ap.add_argument("--side", default="aligned", choices=["aligned", "base"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = json.loads((RESULTS_DIR / args.run / "raw_results.json").read_text())
    records = raw[args.side]["attacked"]

    rng = random.Random(args.seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    picked = idx[: args.n]

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "sample.csv"

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "answer", "classifier_severity",
                    "classifier_refusal_type", "my_label"])
        for i in picked:
            r = records[i]
            # Flatten newlines so the CSV stays one row per record.
            answer = " ".join(str(r.get("answer", "")).split())
            w.writerow([
                i,
                " ".join(str(r.get("query", "")).split()),
                answer,
                r.get("severity", ""),
                r.get("refusal_type", ""),
                "",  # you fill this in
            ])

    # Report the class balance so you know what you are labelling.
    sev = [records[i].get("severity") for i in picked]
    n_denial = sum(1 for s in sev if s == FULL_DENIAL)
    print(f"Wrote {out_path}")
    print(f"  {args.n} records from {args.run} [{args.side}]")
    print(f"  classifier says full denial: {n_denial}/{args.n}")
    print()
    print("Now open validation/sample.csv and fill my_label with: refusal | answer")
    print("Label from the ANSWER text alone. Do not look at classifier_severity.")


if __name__ == "__main__":
    main()
