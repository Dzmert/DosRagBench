"""Score the hand-labelled sample against the refusal classifier.

Usage:
    python scripts/score_validation.py

Reads validation/sample.csv (my_label filled in) and reports agreement,
false positive rate, false negative rate and Cohen's kappa.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "validation" / "sample.csv"

FULL_DENIAL = 3


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    rows = list(csv.DictReader(open(SAMPLE)))
    labelled = [r for r in rows if r["my_label"].strip()]

    if not labelled:
        print(f"No labels found in {SAMPLE}. Fill the my_label column first.")
        return
    if len(labelled) < len(rows):
        print(f"Warning: {len(rows) - len(labelled)} rows unlabelled, skipping them.\n")

    tp = fp = tn = fn = 0
    disagreements = []

    for r in labelled:
        clf = int(r["classifier_severity"]) == FULL_DENIAL
        human = r["my_label"].strip().lower().startswith("r")

        if clf and human:
            tp += 1
        elif clf and not human:
            fp += 1
            disagreements.append(("FP", r))
        elif not clf and human:
            fn += 1
            disagreements.append(("FN", r))
        else:
            tn += 1

    n = tp + fp + tn + fn
    agree = tp + tn

    # Cohen's kappa
    po = agree / n
    p_clf = (tp + fp) / n
    p_hum = (tp + fn) / n
    pe = p_clf * p_hum + (1 - p_clf) * (1 - p_hum)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0

    lo, hi = wilson(agree, n)

    print(f"n = {n}\n")
    print("            human:refusal  human:answer")
    print(f"clf:refusal      {tp:>6}        {fp:>6}")
    print(f"clf:answer       {fn:>6}        {tn:>6}")
    print()
    print(f"Agreement        {po*100:.1f}%  [{lo*100:.1f}, {hi*100:.1f}]")
    print(f"Cohen's kappa    {kappa:.3f}")
    if tp + fp:
        print(f"False positive   {fp/(tp+fp)*100:.1f}%  (classifier said refusal, was an answer)")
    if tp + fn:
        print(f"False negative   {fn/(tp+fn)*100:.1f}%  (classifier missed a refusal)")

    if disagreements:
        print(f"\n{len(disagreements)} disagreements:")
        for kind, r in disagreements[:10]:
            print(f"  [{kind}] id={r['id']} sev={r['classifier_severity']} "
                  f"type={r['classifier_refusal_type']}")
            print(f"        {r['answer'][:110]}")
        if len(disagreements) > 10:
            print(f"  ... and {len(disagreements) - 10} more")


if __name__ == "__main__":
    main()
