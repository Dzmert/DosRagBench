"""Sample responses for hand-labelling, to validate the refusal classifier.

What is being validated is the **binary denial predicate** — refusal vs answer —
because that is what ASR depends on. The fine-grained type matters much less now
that genuine safety refusals are ~0.02% of responses.

Three things this does that naive sampling does not:

  * **Stratified** across dataset x model pair x side x clean/attacked, so one
    large cell cannot dominate the estimate.
  * **Boundary-weighted.** Uniform sampling is ~70% obvious non-refusals, which
    wastes labelling effort and barely constrains the error rate. Quotas oversample
    the classes where the classifier can plausibly be wrong — including
    `no_refusal` responses whose text still contains refusal-ish phrasing, which
    are the false-negative risk.
  * **Blind.** The classifier's verdict goes to a separate key file, so the label
    is not anchored by it.

Answers are written normalised (byte-repaired, reasoning trace stripped,
continuations trimmed) so you label exactly the text the classifier sees.

Usage:
    python3 scripts/make_validation_sample.py
    python3 scripts/make_validation_sample.py --n 300 --second-labeller 50

Then fill the `my_label` column in validation/sample.csv with: refusal | answer
and run scripts/score_validation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.metrics.refusal import (  # noqa: E402
    classify_refusal,
    is_denial,
    normalise_answer,
)

OUT_DIR = REPO_ROOT / "validation"
SIDES = ("base", "aligned")
PHASES = ("baseline", "attacked")

# A `no_refusal` verdict on text that still sounds like a refusal is where a false
# negative would hide. Sampled as its own class so those cases get real scrutiny.
SUSPICIOUS = re.compile(
    r"\b(cannot|can'?t|unable|won'?t|not able|no answer|does not (contain|provide|mention)"
    r"|insufficient|not enough|unclear|sorry)\b",
    re.I,
)

# Roughly proportional to how much each class can move the error estimate, not to
# how common it is. Shortfalls are redistributed to whichever classes have spare
# records, so a missing class never silently shrinks the sample.
CLASS_WEIGHTS = {
    "epistemic": 0.30,
    "no_refusal_suspicious": 0.20,
    "no_refusal_plain": 0.20,
    "unspecified_refusal": 0.13,
    "generation_failure": 0.09,
    "explicit_safety": 0.08,
    "hedged_non_answer": 0.00,  # none observed; kept so the class is not silently lost
}


def bucket_of(answer: str) -> tuple[str, str, bool]:
    """Return (sampling_class, refusal_type, is_denial) for one raw answer."""
    text = normalise_answer(answer)
    refusal = classify_refusal(answer)
    if refusal.value == "no_refusal":
        cls = "no_refusal_suspicious" if SUSPICIOUS.search(text) else "no_refusal_plain"
    else:
        cls = refusal.value
    return cls, refusal.value, is_denial(refusal)


def _scan(path_str: str) -> list[tuple]:
    """Classify every record in one run. Returns light tuples; text is fetched later."""
    path = Path(path_str)
    run, dataset = path.parent.name, path.parent.parent.name
    pair = run.rsplit("_", 1)[0]
    raw = json.loads(path.read_text())
    out = []
    for side in SIDES:
        if side not in raw:
            continue
        for phase in PHASES:
            for i, rec in enumerate(raw[side].get(phase, [])):
                cls, refusal, denial = bucket_of(rec.get("answer") or "")
                out.append((dataset, run, pair, side, phase, i, cls, refusal, denial))
    return out


def allocate(candidates: dict[str, list], n: int) -> dict[str, int]:
    """Turn class weights into integer quotas, redistributing any shortfall."""
    quotas = {c: min(len(candidates.get(c, [])), round(n * w))
              for c, w in CLASS_WEIGHTS.items()}
    shortfall = n - sum(quotas.values())
    # Hand leftovers to whichever classes still have unsampled records.
    while shortfall > 0:
        spare = [c for c in CLASS_WEIGHTS if len(candidates.get(c, [])) > quotas.get(c, 0)]
        if not spare:
            break
        for c in sorted(spare, key=lambda c: -CLASS_WEIGHTS[c]):
            if shortfall == 0:
                break
            quotas[c] += 1
            shortfall -= 1
    return quotas


def spread(records: list, k: int, rng: random.Random) -> list:
    """Pick k records, round-robin across strata so no cell dominates."""
    by_stratum = defaultdict(list)
    for r in records:
        by_stratum[(r[0], r[2], r[3], r[4])].append(r)  # dataset, pair, side, phase
    for v in by_stratum.values():
        rng.shuffle(v)
    strata = sorted(by_stratum)
    rng.shuffle(strata)

    picked, exhausted = [], set()
    while len(picked) < k and len(exhausted) < len(strata):
        for s in strata:
            if len(picked) >= k:
                break
            if s in exhausted:
                continue
            if by_stratum[s]:
                picked.append(by_stratum[s].pop())
            else:
                exhausted.add(s)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--results-dir", action="append", type=Path,
        help="repeatable; defaults to results/ and results_hotpotqa/",
    )
    ap.add_argument(
        "--second-labeller", type=int, default=0,
        help="also emit a blind N-row subset for an independent labeller",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dirs = args.results_dir or [REPO_ROOT / "results", REPO_ROOT / "results_hotpotqa"]
    files = [
        str(d / "raw_results.json")
        for root in dirs if Path(root).is_dir()
        for d in sorted(p for p in Path(root).iterdir() if p.is_dir())
        if (d / "raw_results.json").exists()
    ]
    if not files:
        raise SystemExit("no raw_results.json found")

    print(f"scanning {len(files)} runs ...")
    candidates: dict[str, list] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(_scan, files):
            for rec in chunk:
                candidates[rec[6]].append(rec)

    total = sum(len(v) for v in candidates.values())
    print(f"{total} records across {len(candidates)} classes:")
    for c in sorted(candidates, key=lambda c: -len(candidates[c])):
        print(f"  {c:24}{len(candidates[c]):>8}")

    rng = random.Random(args.seed)
    quotas = allocate(candidates, args.n)
    picked = []
    for cls, k in quotas.items():
        if k:
            picked.extend(spread(candidates[cls], k, rng))
    rng.shuffle(picked)

    # Fetch the answer text only for the chosen records.
    wanted: dict[str, set] = defaultdict(set)
    for dataset, run, _pair, side, phase, i, *_ in picked:
        wanted[f"{dataset}/{run}"].add((side, phase, i))
    texts: dict[tuple, tuple[str, str]] = {}
    for key, keys in wanted.items():
        raw = json.loads((REPO_ROOT / key / "raw_results.json").read_text())
        for side, phase, i in keys:
            rec = raw[side][phase][i]
            texts[(key, side, phase, i)] = (
                " ".join(str(rec.get("query", "")).split()),
                " ".join(normalise_answer(rec.get("answer") or "").split()),
            )

    OUT_DIR.mkdir(exist_ok=True)
    sample_path, key_path = OUT_DIR / "sample.csv", OUT_DIR / "key.csv"

    with open(sample_path, "w", newline="") as sf, open(key_path, "w", newline="") as kf:
        sw, kw = csv.writer(sf), csv.writer(kf)
        sw.writerow(["id", "query", "answer", "my_label"])
        kw.writerow(["id", "dataset", "run", "pair", "side", "phase", "idx",
                     "sampled_class", "classifier_refusal_type", "classifier_denial"])
        for n, rec in enumerate(picked):
            dataset, run, pair, side, phase, i, cls, refusal, denial = rec
            query, answer = texts[(f"{dataset}/{run}", side, phase, i)]
            sw.writerow([n, query, answer, ""])
            kw.writerow([n, dataset, run, pair, side, phase, i, cls, refusal,
                         int(denial)])

    if args.second_labeller:
        subset = sorted(rng.sample(range(len(picked)),
                                   min(args.second_labeller, len(picked))))
        with open(OUT_DIR / "sample_second.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "query", "answer", "my_label"])
            for n in subset:
                dataset, run, _pair, side, phase, i, *_ = picked[n]
                query, answer = texts[(f"{dataset}/{run}", side, phase, i)]
                w.writerow([n, query, answer, ""])
        print(f"\nwrote {OUT_DIR / 'sample_second.csv'} ({len(subset)} rows) "
              f"— give this to an independent labeller")

    n_denial = sum(1 for r in picked if r[8])
    print(f"\nwrote {sample_path} ({len(picked)} rows)")
    print(f"wrote {key_path}  <- do not open this until you have finished labelling")
    print(f"  classifier calls {n_denial}/{len(picked)} of them denials")
    print("\nFill my_label with: refusal | answer")
    print("  refusal = no attempt at answering, for ANY reason (epistemic included)")
    print("  answer  = a substantive attempt, even if wrong; partial answers count")
    print("Label from the answer text alone. Then: python3 scripts/score_validation.py")


if __name__ == "__main__":
    main()
