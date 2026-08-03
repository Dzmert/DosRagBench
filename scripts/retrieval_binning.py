"""Does the alignment gap widen as retrieval degrades?

The corrected results reframed the mechanism: genuine safety refusals are ~0.02%
of responses, so what the attacks exploit is epistemic caution — instruction
tuning teaches "answer only from the provided context", and a model holding that
rule refuses whenever the context looks inadequate. Polluting the retrieved set
is therefore an attack on the *appearance* of evidence, not on safety guardrails.

That predicts a dose-response: the worse retrieval already is, the less pollution
it takes to push an aligned model over the line. This tests it without a third
corpus, by binning queries on the retrieval quality they had **before** the
attack and measuring the alignment gap within each bin.

Binning uses the clean-run gold_rank. Using the attacked rank would confound the
independent variable with the treatment.

Two aggregations are reported because they answer different questions:
  * macro — per-run risk_diff averaged over runs. Runs are the independent unit,
    so this is the headline.
  * pooled — every query-instance together. Larger n, but the same query recurs
    once per attack, so the effective sample is smaller than it looks.

Usage:
    python3 scripts/retrieval_binning.py
    python3 scripts/retrieval_binning.py --csv results/retrieval_binning.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.metrics.refusal import classify_refusal, is_denial  # noqa: E402

# Ordered worst-retrieval-last so a rising risk_diff reads as a dose-response.
BINS = [
    ("rank 0", lambda r: r == 0),
    ("rank 1-2", lambda r: r in (1, 2)),
    ("rank 3-4", lambda r: r in (3, 4)),
    ("absent", lambda r: r < 0),
]
SIDES = ("base", "aligned")


def bin_of(rank: int) -> str:
    for name, test in BINS:
        if test(rank):
            return name
    return "absent"


def analyse(path_str: str) -> dict | None:
    """Per-bin denial counts for one run, both sides."""
    path = Path(path_str)
    run, dataset = path.parent.name, path.parent.parent.name
    raw = json.loads(path.read_text())
    if not all(s in raw for s in SIDES):
        return None

    # Skip runs whose retrieval failed outright; they are excluded everywhere else.
    base_clean = raw["base"]["baseline"]
    if not any(r.get("gold_in_topk") for r in base_clean):
        return None

    out: dict = {"dataset": dataset, "run": run, "pair": run.rsplit("_", 1)[0], "bins": {}}
    for side in SIDES:
        clean, attacked = raw[side]["baseline"], raw[side]["attacked"]
        per_bin = defaultdict(
            lambda: {"n": 0, "clean_denied": 0, "answerable": 0, "broken": 0,
                     "attacked_denied": 0}
        )
        for c, a in zip(clean, attacked):
            b = per_bin[bin_of(int(c.get("gold_rank", -1)))]
            b["n"] += 1
            c_den = is_denial(classify_refusal(c.get("answer") or ""))
            if c_den:
                b["clean_denied"] += 1
                continue  # not answerable, so it cannot be attributed to the attack
            b["answerable"] += 1
            if is_denial(classify_refusal(a.get("answer") or "")):
                b["broken"] += 1
        # Unconditional attacked denial, counted over every query in the bin.
        # Attributable ASR conditions on clean-answerable, and that denominator
        # shrinks sharply as retrieval degrades — by the "absent" bin the survivors
        # are the queries the model was willing to answer despite bad retrieval,
        # i.e. an increasingly self-selected and unusually robust subset. The
        # unconditional rate has the opposite bias (it credits the attack for
        # pre-existing refusal) but no selection effect, so reading both bounds
        # the truth.
        for c, a in zip(clean, attacked):
            b = per_bin[bin_of(int(c.get("gold_rank", -1)))]
            if is_denial(classify_refusal(a.get("answer") or "")):
                b["attacked_denied"] = b.get("attacked_denied", 0) + 1
        out["bins"][side] = {k: dict(v) for k, v in per_bin.items()}
    return out


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--min-answerable", type=int, default=25,
                    help="drop a run/bin cell below this many answerable queries")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = [
        str(d / "raw_results.json")
        for root in (REPO_ROOT / "results", REPO_ROOT / "results_hotpotqa")
        if root.is_dir()
        for d in sorted(p for p in root.iterdir() if p.is_dir())
        if (d / "raw_results.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = [r for r in pool.map(analyse, files) if r]

    print(f"{len(runs)} runs (5 zero-recall runs excluded)\n")
    bin_names = [b[0] for b in BINS]
    rows = []

    for label, keep in (
        ("NQ (excl. llama-r1)", lambda r: r["dataset"] == "results" and "llama-r1" not in r["pair"]),
        ("NQ llama-r1 only", lambda r: r["dataset"] == "results" and "llama-r1" in r["pair"]),
        ("HotpotQA", lambda r: r["dataset"] == "results_hotpotqa"),
    ):
        group = [r for r in runs if keep(r)]
        if not group:
            continue
        print(f"=== {label}  ({len(group)} runs)")
        print(f"{'bin':12}{'runs':>6}{'macro rd':>11}{'pooled rd':>11}"
              f"{'aln clean':>11}{'n_ans':>8}"
              f"{'| uncond. attacked denial: base':>32}{'aligned':>9}{'gap':>8}")

        for bn in bin_names:
            per_run, pooled = [], {s: {"answerable": 0, "broken": 0, "n": 0,
                                       "clean_denied": 0, "attacked_denied": 0}
                                   for s in SIDES}
            for r in group:
                cells = {s: r["bins"][s].get(bn) for s in SIDES}
                if not all(cells.values()):
                    continue
                for s in SIDES:
                    for k in pooled[s]:
                        pooled[s][k] += cells[s][k]
                if all(cells[s]["answerable"] >= args.min_answerable for s in SIDES):
                    per_run.append(
                        cells["aligned"]["broken"] / cells["aligned"]["answerable"]
                        - cells["base"]["broken"] / cells["base"]["answerable"]
                    )
            macro = mean(per_run) if per_run else None
            pooled_rd = None
            if pooled["aligned"]["answerable"] and pooled["base"]["answerable"]:
                pooled_rd = (pooled["aligned"]["broken"] / pooled["aligned"]["answerable"]
                             - pooled["base"]["broken"] / pooled["base"]["answerable"])
            aln_clean = rate(pooled["aligned"]["clean_denied"], pooled["aligned"]["n"])
            unc = {s: rate(pooled[s]["attacked_denied"], pooled[s]["n"]) for s in SIDES}
            unc_gap = (unc["aligned"] - unc["base"]) if all(v is not None for v in unc.values()) else None
            fmt = lambda v: f"{v:+.3f}" if v is not None else "  n/a"
            pct = lambda v: f"{v:.3f}" if v is not None else "n/a"
            print(f"{bn:12}{len(per_run):>6}{fmt(macro):>11}{fmt(pooled_rd):>11}"
                  f"{pct(aln_clean):>11}{pooled['aligned']['answerable']:>8}"
                  f"{pct(unc['base']):>32}{pct(unc['aligned']):>9}{fmt(unc_gap):>8}")
            rows.append({
                "group": label, "bin": bn, "n_runs": len(per_run),
                "macro_risk_diff": None if macro is None else round(macro, 4),
                "macro_sd": round(stdev(per_run), 4) if len(per_run) > 1 else None,
                "pooled_risk_diff": None if pooled_rd is None else round(pooled_rd, 4),
                "aligned_clean_denial": None if aln_clean is None else round(aln_clean, 4),
                "pooled_answerable_aligned": pooled["aligned"]["answerable"],
                "uncond_attacked_denial_base": None if unc["base"] is None else round(unc["base"], 4),
                "uncond_attacked_denial_aligned": None if unc["aligned"] is None else round(unc["aligned"], 4),
                "uncond_gap": None if unc_gap is None else round(unc_gap, 4),
                "pooled_n_queries": pooled["aligned"]["n"],
            })

        macros = [r["macro_risk_diff"] for r in rows
                  if r["group"] == label and r["macro_risk_diff"] is not None]
        if len(macros) > 1:
            trend = "rises" if macros[-1] > macros[0] else "falls"
            monotone = all(b >= a for a, b in zip(macros, macros[1:]))
            print(f"  -> risk_diff {trend} {macros[0]:+.3f} -> {macros[-1]:+.3f}"
                  f"  ({'monotonic' if monotone else 'not monotonic'})\n")

    if args.csv:
        out = REPO_ROOT / args.csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
