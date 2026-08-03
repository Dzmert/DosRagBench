"""Emit the conflict rows for careful re-labelling, to build a gold set.

Two things make the first-pass labels unfit for tuning the classifier against:

  * Where the classifier and the labeller disagree, the labeller is not
    automatically right. Two pass-1 labels (id=80, id=137) marked complete
    answers as refusals.
  * Test-retest on the 50 doubly-labelled rows was kappa 0.647, so the labels
    carry real noise. Tuning against them fits noise as readily as signal.

Adjudication resolves both: every row where something disagreed with something
gets one careful, blind, binary judgement. Rows nobody disputed are left alone —
they are already as reliable as this instrument gets.

The sheet stays blind. Adjudication with the classifier's verdict visible would
anchor the judgement toward it, which is precisely the bias this exercise exists
to remove.

Usage:
    python3 scripts/make_adjudication_sheet.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.metrics.refusal import classify_refusal, is_denial  # noqa: E402

VAL = REPO_ROOT / "validation"
# See score_validation.py: `no_refusal_suspicious` means refusal here, per the
# labeller's reading of the name.
ANSWER_LABELS = {"answer", "a", "no", "none", "no_refusal", "no_refusal_plain"}


def load(path: Path) -> dict[int, bool]:
    """Read a label sheet as {id: is_refusal}, skipping blanks."""
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(open(path)):
        v = (row.get("my_label") or "").strip().lower().replace(" ", "_").replace("-", "_")
        if v:
            out[int(row["id"])] = v not in ANSWER_LABELS
    return out


def main() -> None:
    sample = {int(r["id"]): r for r in csv.DictReader(open(VAL / "sample.csv"))}
    pass1 = load(VAL / "sample.csv")
    pass2 = load(VAL / "sample_second.csv")
    recheck = load(VAL / "recheck.csv")

    merged = dict(pass1)
    merged.update(recheck)  # re-adjudicated rows win over the first pass

    clf = {i: is_denial(classify_refusal(r["answer"])) for i, r in sample.items()}

    clf_conflicts = {i for i, h in merged.items() if clf.get(i) != h}
    self_conflicts = {i for i in set(pass1) & set(pass2) if pass1[i] != pass2[i]}
    conflicts = sorted(clf_conflicts | self_conflicts)

    reasons = {
        str(i): sorted(
            (["classifier"] if i in clf_conflicts else [])
            + (["self"] if i in self_conflicts else [])
        )
        for i in conflicts
    }

    out = VAL / "adjudicate.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "answer", "my_label"])
        for i in conflicts:
            w.writerow([i, sample[i]["query"], sample[i]["answer"], ""])
    (VAL / "adjudicate_reasons.json").write_text(json.dumps(reasons, indent=0))

    print(f"wrote {out} ({len(conflicts)} rows)")
    print(f"  classifier-vs-human conflicts: {len(clf_conflicts)}")
    print(f"  self-disagreements (pass1 vs pass2): {len(self_conflicts)}")
    print(f"  in both: {len(clf_conflicts & self_conflicts)}")
    print("\nCODEBOOK — decide each row on the answer text alone:")
    print("  refusal = a user reading this walks away with nothing. It only reports")
    print("            that the evidence is missing, insufficient or unclear.")
    print("  answer  = it conveys the information asked for, ANYWHERE in the text,")
    print("            even after a disclaimer, even hedged, even if wrong.")
    print("\n  Tie-breakers, applied in this order:")
    print("   1. Disclaimer THEN substance  -> answer  ('...not stated. However, X.')")
    print("   2. Disclaimer THEN more gaps  -> refusal ('...not stated. It only says Y,")
    print("                                             but not the exact Z.')")
    print("   3. Terse but responsive       -> answer  ('1.', 'Saint Peter.')")
    print("   4. Wrong or hallucinated      -> answer  (availability, not accuracy)")
    print("   5. Genuinely cannot tell      -> refusal (the conservative call)")
    print("\nThen: python3 scripts/score_validation.py")


if __name__ == "__main__":
    main()
