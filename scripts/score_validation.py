"""Score the hand-labelled sample against the refusal classifier.

The classifier's verdict is recomputed **live** from the stored answer text rather
than read out of the sample file. That way the labels stay valid when refusal.py
changes: re-run this script and you get the current classifier's error rate against
the same human judgements, with no relabelling.

Reports agreement with a Wilson interval, false positive and false negative rates,
Cohen's kappa, a per-class breakdown showing where the errors concentrate, and —
if validation/sample_second.csv is filled in — inter-annotator agreement.

Usage:
    python3 scripts/score_validation.py
    python3 scripts/score_validation.py --show 25
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.metrics.refusal import (  # noqa: E402
    classify_refusal,
    is_denial,
    normalise_answer,
)

VAL_DIR = REPO_ROOT / "validation"
SAMPLE = VAL_DIR / "sample.csv"
KEY = VAL_DIR / "key.csv"
SECOND = VAL_DIR / "sample_second.csv"


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def kappa_of(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two boolean label vectors."""
    n = len(a)
    if not n:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def parse_label(raw: str) -> bool | None:
    """'refusal' -> True, 'answer' -> False, anything else -> None."""
    v = (raw or "").strip().lower()
    if v.startswith("r"):
        return True
    if v.startswith("a"):
        return False
    return None


def load_labels(path: Path) -> dict[int, bool]:
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(open(path)):
        label = parse_label(row.get("my_label", ""))
        if label is not None:
            out[int(row["id"])] = label
    return out


def classifier_verdicts(key_rows: list[dict]) -> dict[int, tuple[bool, str]]:
    """Reclassify each sampled record from its raw answer, live."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in key_rows:
        by_run[f"{r['dataset']}/{r['run']}"].append(r)

    out: dict[int, tuple[bool, str]] = {}
    for run_key, rows in by_run.items():
        raw = json.loads((REPO_ROOT / run_key / "raw_results.json").read_text())
        for r in rows:
            rec = raw[r["side"]][r["phase"]][int(r["idx"])]
            refusal = classify_refusal(rec.get("answer") or "")
            out[int(r["id"])] = (is_denial(refusal), refusal.value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=12, help="disagreements to print")
    args = ap.parse_args()

    if not SAMPLE.exists():
        raise SystemExit(f"{SAMPLE} not found — run make_validation_sample.py first")
    if not KEY.exists():
        raise SystemExit(f"{KEY} not found — regenerate the sample")

    sample_rows = list(csv.DictReader(open(SAMPLE)))
    key_rows = list(csv.DictReader(open(KEY)))
    human = load_labels(SAMPLE)

    if not human:
        raise SystemExit(f"No labels in {SAMPLE}. Fill the my_label column first.")
    if len(human) < len(sample_rows):
        print(f"Warning: {len(sample_rows) - len(human)} of {len(sample_rows)} rows "
              f"unlabelled; scoring the rest.\n")

    key_by_id = {int(r["id"]): r for r in key_rows}
    text_by_id = {int(r["id"]): r["answer"] for r in sample_rows}
    verdicts = classifier_verdicts([key_by_id[i] for i in human if i in key_by_id])

    tp = fp = tn = fn = 0
    disagreements = []
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [errors, total]
    clf_vec, hum_vec = [], []

    for i, h in sorted(human.items()):
        if i not in verdicts:
            continue
        c, ctype = verdicts[i]
        clf_vec.append(c)
        hum_vec.append(h)
        cls = key_by_id[i]["sampled_class"]
        per_class[cls][1] += 1
        if c and h:
            tp += 1
        elif c and not h:
            fp += 1
            per_class[cls][0] += 1
            disagreements.append(("FP", i, ctype))
        elif not c and h:
            fn += 1
            per_class[cls][0] += 1
            disagreements.append(("FN", i, ctype))
        else:
            tn += 1

    n = tp + fp + tn + fn
    if not n:
        raise SystemExit("no labelled rows could be matched to the key")
    agree = tp + tn
    lo, hi = wilson(agree, n)

    print(f"n = {n}\n")
    print("            human:refusal  human:answer")
    print(f"clf:refusal      {tp:>6}        {fp:>6}")
    print(f"clf:answer       {fn:>6}        {tn:>6}")
    print()
    print(f"Agreement        {agree / n * 100:.1f}%  [{lo * 100:.1f}, {hi * 100:.1f}]")
    print(f"Cohen's kappa    {kappa_of(clf_vec, hum_vec):.3f}   (target > 0.8)")
    if tp + fp:
        print(f"False positive   {fp / (tp + fp) * 100:.1f}%  "
              f"(classifier said refusal, was an answer)")
    if tp + fn:
        print(f"False negative   {fn / (tp + fn) * 100:.1f}%  "
              f"(classifier missed a refusal)")

    print(f"\n{'sampled class':26}{'errors':>8}{'n':>6}{'rate':>8}")
    for cls in sorted(per_class, key=lambda c: -per_class[c][0]):
        err, tot = per_class[cls]
        print(f"{cls:26}{err:>8}{tot:>6}{err / tot * 100:>7.1f}%")

    second = load_labels(SECOND)
    if second:
        shared = sorted(set(second) & set(human))
        if shared:
            k = kappa_of([human[i] for i in shared], [second[i] for i in shared])
            same = sum(1 for i in shared if human[i] == second[i])
            print(f"\nInter-annotator (n={len(shared)}): "
                  f"agreement {same / len(shared) * 100:.1f}%, kappa {k:.3f}")
    elif SECOND.exists():
        print(f"\n{SECOND.name} exists but is unlabelled — "
              f"inter-annotator agreement not computed.")

    if disagreements:
        print(f"\n{len(disagreements)} disagreements:")
        for kind, i, ctype in disagreements[: args.show]:
            k = key_by_id[i]
            print(f"  [{kind}] id={i} {k['dataset']}/{k['run']} {k['side']}/{k['phase']} "
                  f"clf={ctype}")
            print(f"        {text_by_id[i][:150]}")
        if len(disagreements) > args.show:
            print(f"  ... and {len(disagreements) - args.show} more "
                  f"(--show {len(disagreements)} for all)")


if __name__ == "__main__":
    main()
