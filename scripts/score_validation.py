"""Score the hand-labelled sample against the refusal classifier.

The classifier's verdict is recomputed **live** from the stored answer text rather
than read out of the sample file. That way the labels stay valid when refusal.py
changes: re-run this script and you get the current classifier's error rate against
the same human judgements, with no relabelling.

Reports agreement with a Wilson interval, false positive and false negative rates,
Cohen's kappa, a per-class breakdown showing where the errors concentrate, and —
if validation/sample_second.csv is filled in — inter-annotator agreement.

Two further figures are quoted in docs/findings_summary.md and produced here so
that they are reproducible rather than computed by hand:

* the dev/holdout split (validation/split.json), since the holdout kappa is the
  figure to quote for instrument quality;
* the population-reweighted error rate. The sample is deliberately boundary-
  weighted, so its raw agreement is pessimistic; reweighting the per-class error
  rates by how common each class actually is gives the corpus-level error.

Usage:
    python3 scripts/score_validation.py
    python3 scripts/score_validation.py --show 25
    python3 scripts/score_validation.py --no-reweight   # skips a ~20 s corpus scan
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
# An independent annotator's pass over the same 50 rows. Kept separate from
# SECOND, which is the primary labeller's own second pass -- that measures
# test-retest, not inter-annotator agreement, and the two must not be conflated.
ANNOTATOR_B = VAL_DIR / "annotator_b.csv"
# Re-adjudicated rows. The first labelling pass applied a "was the model behaving
# correctly" reading, under which reporting missing evidence is an answer. The
# agreed construct is availability — the user asked and got nothing — so the rows
# whose text reports absence were relabelled against the clarified rule. These
# override sample.csv wherever both carry a label.
RECHECK = VAL_DIR / "recheck.csv"
# Final adjudication of every row where the classifier and the labeller, or the
# labeller's two passes, disagreed. Highest precedence: sample < recheck < this.
ADJUDICATE = VAL_DIR / "adjudicate.csv"
# id -> "dev" | "holdout". The dev rows were used to diagnose classifier gaps, so
# only the holdout figure is an out-of-sample estimate of instrument quality.
SPLIT = VAL_DIR / "split.json"

# Roots scanned for the population class frequencies used to reweight the error
# rate. Same roots the rest of the pipeline uses.
CORPUS_ROOTS = ("results", "results_hotpotqa")


def population_class_counts() -> tuple[dict[str, int], int, int]:
    """Count every stored answer by sampling class. Returns (counts, total, runs).

    Reuses `bucket_of` from make_validation_sample so the class rule cannot drift
    between the sheet that was sampled and the weights applied to it.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from make_validation_sample import bucket_of  # noqa: E402

    counts: dict[str, int] = defaultdict(int)
    total = runs = 0
    for root in CORPUS_ROOTS:
        for path in sorted((REPO_ROOT / root).glob("*/raw_results.json")):
            raw = json.loads(path.read_text())
            runs += 1
            for side in ("base", "aligned"):
                for phase in ("baseline", "attacked"):
                    for rec in (raw.get(side) or {}).get(phase, []) or []:
                        cls, _, _ = bucket_of(rec.get("answer") or "")
                        counts[cls] += 1
                        total += 1
    return counts, total, runs


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
    ap.add_argument("--no-reweight", action="store_true",
                    help="skip the population reweighting (saves a ~20 s corpus scan)")
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
    clf_vec, hum_vec, scored_ids = [], [], []

    for i, h in sorted(human.items()):
        if i not in verdicts:
            continue
        c, ctype = verdicts[i]
        clf_vec.append(c)
        hum_vec.append(h)
        scored_ids.append(i)
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

    # Out-of-sample estimate. The dev rows informed the classifier's patterns, so
    # only the holdout number is an unbiased read on instrument quality.
    if SPLIT.exists():
        split = json.loads(SPLIT.read_text())
        print(f"\n{'split':10}{'n':>5}{'agreement':>12}{'kappa':>8}")
        for part in ("dev", "holdout"):
            idx = [j for j, i in enumerate(scored_ids) if split.get(str(i)) == part]
            if not idx:
                continue
            h = [hum_vec[j] for j in idx]
            c = [clf_vec[j] for j in idx]
            same = sum(1 for x, y in zip(h, c) if x == y)
            print(f"{part:10}{len(idx):>5}{same / len(idx) * 100:>11.1f}%"
                  f"{kappa_of(c, h):>8.3f}")
        print("           quote the holdout figure for instrument quality")
    elif not args.no_reweight:
        print(f"\n{SPLIT.name} not found — no dev/holdout breakdown.")

    # The sample oversamples the decision boundary on purpose, so its raw error
    # rate is not the corpus error rate. Reweight by how common each class is.
    if not args.no_reweight:
        pop, total_recs, n_runs = population_class_counts()
        print(f"\nPopulation reweighting over {total_recs:,} records in {n_runs} runs")
        print(f"{'sampled class':26}{'pop share':>11}{'sample n':>10}"
              f"{'err rate':>10}{'contrib':>10}")
        weighted = 0.0
        for cls in sorted(pop, key=lambda c: -pop[c]):
            share = pop[cls] / total_recs
            err, tot = per_class.get(cls, [0, 0])
            rate = err / tot if tot else 0.0
            weighted += share * rate
            print(f"{cls:26}{share * 100:>10.2f}%{tot:>10}"
                  f"{rate * 100:>9.1f}%{share * rate * 100:>9.3f}pp")
        unsampled = [c for c in pop if c not in per_class]
        print(f"\nReweighted error {weighted * 100:.2f}%   "
              f"agreement {100 - weighted * 100:.2f}%")
        if unsampled:
            print(f"  note: {', '.join(sorted(unsampled))} carry no sampled rows and "
                  f"are counted as error-free")

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

    other, _, _ = load_labels(ANNOTATOR_B)
    if other:
        shared = sorted(set(other) & set(human))
        if shared:
            k = kappa_of([human[i] for i in shared], [other[i] for i in shared])
            same = sum(1 for i in shared if human[i] == other[i])
            print(f"\nINTER-ANNOTATOR (n={len(shared)}): "
                  f"agreement {same / len(shared) * 100:.1f}%, kappa {k:.3f}")
            clf_k = kappa_of([verdicts[i][0] for i in shared if i in verdicts],
                             [other[i] for i in shared if i in verdicts])
            print(f"  classifier vs annotator B on the same rows: kappa {clf_k:.3f}")
    elif ANNOTATOR_B.exists():
        print(f"\n{ANNOTATOR_B.name} exists but is unlabelled — "
              f"no independent inter-annotator figure yet.")

    second, _, _ = load_labels(SECOND)
    if second:
        # Compare the two ORIGINAL passes. Scoring pass 2 against the adjudicated
        # labels would inflate this: adjudication already corrected part of pass 1,
        # so it would no longer measure how consistent the labeller was with
        # themselves.
        pass1, _, _ = load_labels(SAMPLE)
        shared = sorted(set(second) & set(pass1))
        if shared:
            k = kappa_of([pass1[i] for i in shared], [second[i] for i in shared])
            same = sum(1 for i in shared if pass1[i] == second[i])
            print(f"\nTest-retest, same labeller (n={len(shared)}): "
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
