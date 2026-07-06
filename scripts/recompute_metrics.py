"""Recompute metrics.json from persisted raw_results.json — no GPU required.

All per-query model outputs (answer, severity, latencies, tokens, gold_in_topk)
are already saved in each results/<run>/raw_results.json, so the six-metric
framework can be re-derived offline whenever metrics.py changes. This is what you
run after editing compute_metrics (e.g. the attack-attributable ASR/GDS fix)
instead of re-running the models.

Usage:
    python scripts/recompute_metrics.py            # all runs
    python scripts/recompute_metrics.py --pair llama-3.1-8b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.metrics import QueryResult, compute_metrics
from dosragbench.metrics.refusal import RefusalType, SeverityLevel
from dosragbench.utils.config import RESULTS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def query_result_from_dict(d: dict) -> QueryResult:
    """Rebuild a QueryResult from its persisted to_dict() form."""
    return QueryResult(
        query=d["query"],
        answer=d["answer"],
        retrieval_latency_s=d["retrieval_latency_s"],
        generation_latency_s=d["generation_latency_s"],
        total_latency_s=d["total_latency_s"],
        output_tokens=d["output_tokens"],
        input_tokens=d["input_tokens"],
        retrieved_adversarial_count=d.get("retrieved_adversarial_count", 0),
        gold_in_topk=d.get("gold_in_topk", False),
        gold_rank=d.get("gold_rank", -1),
        refusal_type=RefusalType(d["refusal_type"]),
        severity=SeverityLevel(int(d["severity"])),
    )


def recompute_run(run_dir: Path) -> bool:
    raw_path = run_dir / "raw_results.json"
    if not raw_path.exists():
        return False
    with open(raw_path) as f:
        raw = json.load(f)

    # Preserve attack_category from the existing metrics.json if available;
    # fall back to the run-dir suffix after the model-pair prefix.
    existing = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)

    metrics_reports: dict[str, dict] = {}
    for side_name, side in raw.items():
        if not isinstance(side, dict) or "attacked" not in side:
            continue
        attacked = [query_result_from_dict(r) for r in side["attacked"]]
        baseline = [query_result_from_dict(r) for r in side["baseline"]]
        attack_category = (
            existing.get(side_name, {}).get("attack_category")
            or raw.get("attack_category")
            or run_dir.name.split("_", 1)[-1]
        )
        model_name = side.get("model_name", side_name)
        report = compute_metrics(attacked, baseline, model_name, attack_category)
        metrics_reports[side_name] = report.to_dict()

    if not metrics_reports:
        logger.warning(f"{run_dir.name}: no side with attacked/baseline; skipped")
        return False

    with open(metrics_path, "w") as f:
        json.dump(metrics_reports, f, indent=2)

    for side_name, m in metrics_reports.items():
        logger.info(
            f"{run_dir.name}/{side_name}: "
            f"ASR {m['asr']:.3f} (abs {m['asr_absolute']:.3f}, "
            f"clean-floor {m['baseline_denial_rate']:.3f}, n_ans {m['num_answerable']}) | "
            f"GDS {m['gds']:.3f} (abs {m['gds_absolute']:.3f})"
        )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default=None, help="Filter to run dirs starting with this prefix")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    run_dirs = sorted(d for d in args.results_dir.iterdir() if d.is_dir())
    if args.pair:
        run_dirs = [d for d in run_dirs if d.name.startswith(args.pair)]

    n = sum(1 for d in run_dirs if recompute_run(d))
    logger.info(f"Recomputed metrics.json for {n} run(s)")


if __name__ == "__main__":
    main()
