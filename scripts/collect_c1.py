"""Collect C1 latency/displacement sweep results into a table and plots.

Reads every results/**/c1_latency*.json produced by run_attack.py --c1-latency
and assembles them into:
  - a printed summary table (sorted by KB size then pollution rate)
  - results/c1_summary.csv
  - results/c1_pollution_curve.png (displacement + latency vs pollution)

Handles multiple runs of the same (kb_size, pollution_rate) by averaging them
and reporting the number of runs, so your repeated stability runs are used
rather than clobbered.

Usage:
    python scripts/collect_c1.py
    python scripts/collect_c1.py --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_reports(results_dir: Path) -> list[dict]:
    """Load every c1_latency*.json under results_dir."""
    pattern = str(results_dir / "**" / "c1_latency*.json")
    reports = []
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_path"] = path
            reports.append(data)
        except Exception as exc:
            logger.warning(f"Skipping {path}: {exc}")
    return reports


def aggregate(reports: list[dict]) -> list[dict]:
    """Group by (kb_size, pollution_rate), averaging repeated runs."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in reports:
        key = (r.get("kb_size"), r.get("pollution_rate"))
        groups[key].append(r)

    rows = []
    for (kb_size, pr), runs in groups.items():
        def col(name):
            vals = [x[name] for x in runs if name in x and x[name] is not None]
            return vals

        lir_vals = col("retrieval_lir_mean")
        evic_vals = col("gold_eviction_rate")
        adv_vals = col("mean_adversarial_in_topk")

        rows.append({
            "kb_size": kb_size,
            "pollution_rate": pr,
            "n_runs": len(runs),
            "adversarial_docs": runs[0].get("num_adversarial_docs"),
            "lir_mean": round(mean(lir_vals), 3) if lir_vals else None,
            "lir_std": round(stdev(lir_vals), 3) if len(lir_vals) > 1 else 0.0,
            "eviction_rate": round(mean(evic_vals), 3) if evic_vals else None,
            "adv_in_topk": round(mean(adv_vals), 3) if adv_vals else None,
        })

    rows.sort(key=lambda x: (x["kb_size"] or 0, x["pollution_rate"] or 0))
    return rows


def print_table(rows: list[dict]) -> None:
    print()
    print("=" * 92)
    print(f"{'KB size':>9} {'Pollut.':>8} {'Runs':>5} {'AdvDocs':>8} "
          f"{'LIR':>7} {'±std':>6} {'Eviction':>9} {'Adv@k':>7}")
    print("-" * 92)
    for r in rows:
        print(f"{r['kb_size']:>9} {r['pollution_rate']:>8} {r['n_runs']:>5} "
              f"{str(r['adversarial_docs']):>8} "
              f"{str(r['lir_mean']):>7} {str(r['lir_std']):>6} "
              f"{str(r['eviction_rate']):>9} {str(r['adv_in_topk']):>7}")
    print("=" * 92)
    print()


def write_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {out_path}")


def make_plot(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot. pip install matplotlib")
        return

    # One line per KB size, x = pollution rate
    by_kb: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_kb[r["kb_size"]].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for kb_size, group in sorted(by_kb.items()):
        group = sorted(group, key=lambda x: x["pollution_rate"])
        prs = [g["pollution_rate"] for g in group]
        evic = [g["eviction_rate"] for g in group]
        lir = [g["lir_mean"] for g in group]
        label = f"KB={kb_size:,}"
        ax1.plot(prs, evic, marker="o", label=label)
        ax2.plot(prs, lir, marker="s", label=label)

    ax1.set_xlabel("Pollution rate")
    ax1.set_ylabel("Gold-document eviction rate")
    ax1.set_title("C1 Displacement vs Pollution")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Pollution rate")
    ax2.set_ylabel("Retrieval latency inflation ratio (LIR)")
    ax2.set_title("C1 Latency vs Pollution")
    ax2.axhline(1.0, color="red", lw=0.8, ls="--", label="no effect (LIR=1)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    reports = find_reports(args.results_dir)
    if not reports:
        logger.error(f"No c1_latency*.json files found under {args.results_dir}")
        return
    logger.info(f"Found {len(reports)} C1 report file(s)")

    rows = aggregate(reports)
    print_table(rows)
    write_csv(rows, args.results_dir / "c1_summary.csv")
    make_plot(rows, args.results_dir / "c1_pollution_curve.png")


if __name__ == "__main__":
    main()
