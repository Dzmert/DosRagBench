"""Recompute metrics from persisted raw_results.json — no GPU required.

All per-query model outputs (answer, latencies, tokens, gold_in_topk) are already
saved in each <results-dir>/<run>/raw_results.json, so the six-metric framework can
be re-derived offline whenever metrics.py or refusal.py changes.

By default the refusal type and severity are **reclassified from the answer text**
rather than read back from the stored labels. That matters: three defects found on
2026-08-02 were all in classification, not in the model outputs —

  1. `explicit_safety` captured epistemic refusals (a regex ordering accident
     decided whether a response counted as a full denial),
  2. DeepSeek R1 answers were stored as byte-level BPE token strings, so no regex
     matched and all 13 `llama-r1-8b` runs scored zero refusals,
  3. base-model few-shot continuations appended hallucinated Q&A turns, one of
     which often refused, condemning otherwise-correct answers.

Pass --keep-stored-labels to reproduce the superseded figures instead.

Output goes to metrics_v2.json alongside the original, so previously-reported
numbers stay traceable. Use --in-place to overwrite metrics.json.

Usage:
    python3 scripts/recompute_metrics.py
    python3 scripts/recompute_metrics.py --pair llama-3.1-8b
    python3 scripts/recompute_metrics.py --csv results/recompute_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.metrics import QueryResult, compute_metrics
from dosragbench.metrics.refusal import (
    RefusalType,
    SeverityLevel,
    classify_refusal,
    classify_severity,
)
from dosragbench.utils.config import RESULTS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def query_result_from_dict(
    d: dict, baseline_latency: float | None = None, reclassify: bool = True
) -> QueryResult:
    """Rebuild a QueryResult from its persisted to_dict() form.

    Everything but refusal_type and severity is copied verbatim. Those two are
    re-derived from the raw answer unless the caller asks for the stored labels.
    Severity needs the latency inflation ratio, so attacked records must be paired
    with their clean counterpart's latency.
    """
    answer = d.get("answer") or ""
    total = float(d.get("total_latency_s") or 0.0)

    if reclassify:
        lir = (total / baseline_latency) if baseline_latency else 1.0
        refusal = classify_refusal(answer)
        severity = classify_severity(answer, latency_inflation_ratio=lir)
    else:
        refusal = RefusalType(d["refusal_type"])
        severity = SeverityLevel(int(d["severity"]))

    return QueryResult(
        query=d["query"],
        answer=answer,
        retrieval_latency_s=float(d["retrieval_latency_s"]),
        generation_latency_s=float(d["generation_latency_s"]),
        total_latency_s=total,
        output_tokens=int(d["output_tokens"]),
        input_tokens=int(d["input_tokens"]),
        retrieved_adversarial_count=int(d.get("retrieved_adversarial_count", 0)),
        gold_in_topk=bool(d.get("gold_in_topk", False)),
        gold_rank=int(d.get("gold_rank", -1)),
        refusal_type=refusal,
        severity=severity,
    )


def recompute_run(run_dir: Path, reclassify: bool = True) -> dict | None:
    """Recompute every side of one run. Returns the new metrics dict, or None."""
    raw_path = run_dir / "raw_results.json"
    if not raw_path.exists():
        return None
    with open(raw_path) as f:
        raw = json.load(f)

    # Preserve attack_category from the existing metrics.json if available; fall
    # back to the run-dir suffix. rsplit, not split — the model pair itself
    # contains underscores-adjacent dots (llama-3.1-8b_A1 -> A1).
    existing = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)

    reports: dict[str, dict] = {}
    for side_name, side in raw.items():
        if not isinstance(side, dict) or "attacked" not in side:
            continue

        baseline = [
            query_result_from_dict(r, None, reclassify) for r in side["baseline"]
        ]
        attacked = [
            query_result_from_dict(r, b.total_latency_s or None, reclassify)
            for r, b in zip(side["attacked"], baseline)
        ]

        attack_category = (
            existing.get(side_name, {}).get("attack_category")
            or raw.get("attack_category")
            or run_dir.name.rsplit("_", 1)[-1]
        )
        report = compute_metrics(
            attacked, baseline, side.get("model_name", side_name), attack_category
        )
        d = report.to_dict()

        # compute_metrics only breaks down the attacked side, but the clean refusal
        # profile is what exposed the classifier bug and is needed to read
        # baseline_denial_rate at all.
        clean: dict[str, int] = {}
        for r in baseline:
            clean[r.refusal_type.value] = clean.get(r.refusal_type.value, 0) + 1
        d["baseline_refusal_breakdown"] = clean
        d["hf_id"] = side.get("hf_id")
        d["alignment_level"] = side.get("alignment_level")
        reports[side_name] = d

    return reports or None


def _worker(job: tuple[str, bool, str, bool]) -> tuple[str, str, dict | None, str]:
    run_path, reclassify, out_name, in_place = job
    run_dir = Path(run_path)
    try:
        reports = recompute_run(run_dir, reclassify)
    except Exception as exc:  # one bad run must not kill the sweep
        return run_dir.name, run_dir.parent.name, None, f"{type(exc).__name__}: {exc}"
    if reports:
        target = run_dir / ("metrics.json" if in_place else out_name)
        with open(target, "w") as f:
            json.dump(reports, f, indent=2)
    return run_dir.name, run_dir.parent.name, reports, ""


def _comparison_row(run_dir: Path, dataset: str, new: dict) -> dict:
    old = {}
    path = run_dir / "metrics.json"
    if path.exists():
        try:
            with open(path) as f:
                old = json.load(f)
        except Exception:
            pass

    row = {"dataset": dataset, "run": run_dir.name}
    for side in ("base", "aligned"):
        o, n = old.get(side, {}), new.get(side, {})
        row[f"{side}_asr_old"] = o.get("asr")
        row[f"{side}_asr_new"] = n.get("asr")
        row[f"{side}_bdr_old"] = o.get("baseline_denial_rate")
        row[f"{side}_bdr_new"] = n.get("baseline_denial_rate")
        row[f"{side}_num_answerable_new"] = n.get("num_answerable")

    def diff(a, b):
        return round(a - b, 4) if a is not None and b is not None else None

    row["risk_diff_old"] = diff(row["aligned_asr_old"], row["base_asr_old"])
    row["risk_diff_new"] = diff(row["aligned_asr_new"], row["base_asr_new"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default=None, help="filter to run dirs with this prefix")
    parser.add_argument(
        "--results-dir",
        action="append",
        type=Path,
        help="repeatable; defaults to results/ and results_hotpotqa/",
    )
    parser.add_argument("--out-name", default="metrics_v2.json")
    parser.add_argument("--in-place", action="store_true", help="overwrite metrics.json")
    parser.add_argument(
        "--keep-stored-labels",
        action="store_true",
        help="reuse the stored refusal_type/severity instead of reclassifying",
    )
    parser.add_argument("--csv", default=None, help="write an old-vs-new comparison CSV")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dirs = args.results_dir or [RESULTS_DIR, REPO_ROOT / "results_hotpotqa"]
    jobs = []
    for root in dirs:
        root = Path(root)
        if not root.is_dir():
            logger.warning(f"skipping missing directory: {root}")
            continue
        for run_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            if args.pair and not run_dir.name.startswith(args.pair):
                continue
            if not (run_dir / "raw_results.json").exists():
                continue
            jobs.append(
                (str(run_dir), not args.keep_stored_labels, args.out_name, args.in_place)
            )

    if not jobs:
        raise SystemExit("no runs with raw_results.json found")
    mode = "stored labels" if args.keep_stored_labels else "reclassifying"
    logger.info(f"{len(jobs)} run(s), {mode}, {args.workers} workers")

    rows, failed = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, dataset, reports, err in pool.map(_worker, jobs):
            if err:
                logger.error(f"{name}: {err}")
                failed += 1
                continue
            if not reports:
                logger.warning(f"{name}: no side with attacked/baseline; skipped")
                continue
            run_dir = next(Path(j[0]) for j in jobs if Path(j[0]).name == name)
            row = _comparison_row(run_dir, dataset, reports)
            rows.append(row)
            logger.info(
                f"{dataset}/{name}: risk_diff {row['risk_diff_old']} -> {row['risk_diff_new']}"
            )

    if args.csv and rows:
        out = Path(args.csv)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: (r["dataset"], r["run"]))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"wrote {out}")

    target = "metrics.json" if args.in_place else args.out_name
    logger.info(f"recomputed {len(rows)} run(s) -> {target} ({failed} failed)")


if __name__ == "__main__":
    main()
