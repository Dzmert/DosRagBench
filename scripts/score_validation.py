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
import difflib
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
# Re-adjudicated rows. The first labelling pass applied a "was the model behaving
# correctly" reading, under which reporting missing evidence is an answer. The
# agreed construct is availability — the user asked and got nothing — so the rows
# whose text reports absence were relabelled against the clarified rule. These
# override sample.csv wherever both carry a label.
RECHECK = VAL_DIR / "recheck.csv"
# Final adjudication of every row where the classifier and the labeller, or the
# labeller's two passes, disagreed. Highest precedence: sample < recheck < this.
ADJUDICATE = VAL_DIR / "adjudicate.csv"


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


# A labeller may answer the binary question (refusal / answer) or give the finer
# refusal type. Both are accepted: the type carries strictly more information and
# collapses to the binary unambiguously, so there is no reason to make anyone
# relabel 300 rows for having been more specific than asked.
#
# `no_refusal_suspicious` maps to REFUSAL. The name is ambiguous and the labeller
# read it as "suspicious that this is really a non-refusal" — i.e. it is one —
# rather than the sampling sense of "a non-refusal that merely sounds like one".
# The labeller's reading governs, since they are the one who applied it.
ANSWER_LABELS = {"answer", "a", "no", "none", "no_refusal", "no_refusal_plain"}
REFUSAL_LABELS = {"refusal", "r", "yes", "epistemic", "explicit_safety",
                  "unspecified_refusal", "generation_failure", "hedged_non_answer",
                  "no_refusal_suspicious"}


def parse_label(raw: str) -> tuple[bool, str] | None:
    """Return (is_refusal, canonical_label), or None if unparseable.

    Typos are resolved by closest match — a hand-typed sheet of 300 rows will have
    a few, and silently dropping those rows would bias the estimate.
    """
    v = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not v:
        return None
    if v in ANSWER_LABELS:
        return False, v
    if v in REFUSAL_LABELS:
        return True, v
    near = difflib.get_close_matches(v, list(ANSWER_LABELS | REFUSAL_LABELS), 1, 0.8)
    if near:
        return near[0] in REFUSAL_LABELS, near[0]
    return None


def load_labels(path: Path) -> tuple[dict[int, bool], dict[int, str], list[str]]:
    """Returns (binary labels, canonical type labels, unparseable raw values)."""
    if not path.exists():
        return {}, {}, []
    binary, types, bad = {}, {}, []
    for row in csv.DictReader(open(path)):
        parsed = parse_label(row.get("my_label", ""))
        if parsed is None:
            if (row.get("my_label") or "").strip():
                bad.append(row["my_label"])
            continue
        binary[int(row["id"])], types[int(row["id"])] = parsed[0], parsed[1]
    return binary, types, bad


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
    human, human_types, unparsed = load_labels(SAMPLE)

    recheck, recheck_types, recheck_bad = load_labels(RECHECK)
    if recheck:
        flipped = sum(1 for i, v in recheck.items() if human.get(i) is not None and human[i] != v)
        human.update(recheck)
        human_types.update(recheck_types)
        unparsed += recheck_bad
        print(f"Applied {len(recheck)} re-adjudicated labels from {RECHECK.name} "
              f"({flipped} changed the original verdict).\n")
    elif RECHECK.exists():
        print(f"{RECHECK.name} exists but is unlabelled — "
              f"scoring the original labels only.\n")

    adj, adj_types, adj_bad = load_labels(ADJUDICATE)
    if adj:
        flipped = sum(1 for i, v in adj.items() if human.get(i) is not None and human[i] != v)
        human.update(adj)
        human_types.update(adj_types)
        unparsed += adj_bad
        print(f"Applied {len(adj)} adjudicated labels from {ADJUDICATE.name} "
              f"({flipped} changed the prior verdict).\n")
    elif ADJUDICATE.exists():
        print(f"{ADJUDICATE.name} exists but is unlabelled — "
              f"conflicts not yet adjudicated.\n")

    if not human:
        raise SystemExit(f"No labels in {SAMPLE}. Fill the my_label column first.")
    if unparsed:
        print(f"Warning: {len(unparsed)} unparseable label(s): "
              f"{sorted(set(unparsed))[:5]}\n")
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

    # If the labeller gave refusal types rather than the binary, score that too —
    # it says where the *mechanism* breakdown is wrong, which the binary cannot.
    typed = {i: t for i, t in human_types.items()
             if t not in ("refusal", "answer", "r", "a", "yes", "no")}
    if typed:
        clf_type = {i: verdicts[i][1] for i in typed if i in verdicts}
        norm = {"no_refusal_plain": "no_refusal", "no_refusal_suspicious": "refusal_unspecified_kind"}
        pairs = [(norm.get(typed[i], typed[i]), clf_type[i]) for i in clf_type]
        exact = sum(1 for h, c in pairs if h == c)
        print(f"\nRefusal-type agreement (n={len(pairs)}): {exact / len(pairs) * 100:.1f}%")
        conf: dict[tuple, int] = defaultdict(int)
        for h, c in pairs:
            conf[(h, c)] += 1
        print(f"  {'human':24}{'classifier':24}{'n':>5}")
        for (h, c), cnt in sorted(conf.items(), key=lambda x: -x[1]):
            if cnt >= 3 or h != c:
                print(f"  {h:24}{c:24}{cnt:>5}{'' if h == c else '   <-- differs'}")

    second, _, _ = load_labels(SECOND)
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
