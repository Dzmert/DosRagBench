"""Sensitivity of the cluster filter to its two settings — retrieval only, no LLM.

The defence experiment (§3.15) ran one setting, tau = 0.9 with 2 neighbours.
This sweeps both and reports what each costs on clean queries and what it
removes under attack. Every measure here is a retrieval measure, so no model
is loaded and the sweep runs on a laptop.

Per query, the top-`pool` candidates (with embeddings) are recorded once on the
clean query and once on the attacked query, through the retriever's own
defence hook. Every (tau, min_neighbours) setting is then replayed offline on
the recorded pools with the real ClusterFilter, so the settings are compared on
identical candidates. The attack documents are generated exactly as in
run_attack.py: same attack objects, same seeds, same query order.

Reports, per setting:
  clean     gold in top-5, gold at rank 0, share of queries whose top-5 changed
  attacked  adversarial docs in top-5, gold in top-5 (per attack)

The tau = 0.9 / 2-neighbour / pool-50 cell should match the retrieval columns
of results_defence/defence_report.md; that is the check that the replay is
faithful.

Usage:
    python scripts/sweep_cluster_filter.py
    python scripts/sweep_cluster_filter.py --attacks D2 B3 --num-queries 200
    python scripts/sweep_cluster_filter.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dosragbench.attacks import build_attack  # noqa: E402
from dosragbench.pipeline import HNSWRetriever  # noqa: E402
from dosragbench.pipeline.defence import ClusterFilter, RetrievalDefence  # noqa: E402
from dosragbench.pipeline.retriever import score_gold  # noqa: E402
from dosragbench.utils.config import DATA_DIR, load_attack  # noqa: E402
from run_attack import compute_kb_signature, get_or_build_index, gold_spec, load_dataset  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUT_DIR = REPO_ROOT / "results_defence"
TOP_K = 5


class PoolRecorder(RetrievalDefence):
    """Pass-through defence that keeps the last candidate pool it was shown."""

    name = "record"
    needs_embeddings = True

    def __init__(self):
        self.pool = []

    def apply(self, query, pool):
        self.pool = list(pool)
        return pool


def record_pools(retriever, recorder, query, gold):
    retriever.retrieve(query, top_k=TOP_K, gold_doc_id=gold)
    return recorder.pool


def top_k_metrics(pool, gold, setting):
    """Apply one filter setting to a recorded pool; None means undefended."""
    if setting is not None:
        tau, mn, pool_k = setting
        pool = ClusterFilter(tau=tau, min_neighbours=mn).apply("", pool[:pool_k])
    docs = [c.doc for c in pool[:TOP_K]]
    in_topk, rank, _ = score_gold(docs, gold)
    return docs, in_topk, rank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--attacks", nargs="+", default=["D2", "B3", "A2", "C1"])
    ap.add_argument("--num-queries", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N queries (for a held-out check of a chosen setting)")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.80, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.96, 0.98])
    ap.add_argument("--min-neighbours", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pools", type=int, nargs="+", default=[50])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "cluster_sweep.md")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from an existing <out>.json")
    args = ap.parse_args()

    if args.plot_only:
        plot(json.loads(args.out.with_suffix(".json").read_text(encoding="utf-8")), args.out)
        return

    queries, kb_docs = load_dataset(args.data_dir)
    queries = queries[args.offset: args.offset + args.num_queries]
    retriever = HNSWRetriever(embedder_id=args.embedder, device=args.device)
    get_or_build_index(retriever, kb_docs, args.embedder, compute_kb_signature(kb_docs, args.embedder))
    recorder = PoolRecorder()
    retriever.defences = [recorder]
    retriever.defence_pool_k = max(args.pools)

    settings = [None] + [(t, m, p) for p in args.pools for m in args.min_neighbours for t in args.taus]
    key = lambda s: "undefended" if s is None else f"tau={s[0]:.2f},mn={s[1]},pool={s[2]}"  # noqa: E731

    # Clean pools are attack-independent: record once, score every setting.
    t0 = time.perf_counter()
    retriever.clear_adversarial()
    clean_pools, golds = [], []
    for q in queries:
        golds.append(gold_spec(q))
        clean_pools.append(record_pools(retriever, recorder, q["query"], golds[-1]))
    logger.info(f"Recorded {len(queries)} clean pools in {time.perf_counter() - t0:.0f}s")

    undefended_top = [[d.doc_id for d in top_k_metrics(p, g, None)[0]] for p, g in zip(clean_pools, golds)]
    rows = {}
    for s in settings:
        hits, rank0, changed = [], [], []
        for pool, gold, plain in zip(clean_pools, golds, undefended_top):
            docs, in_topk, rank = top_k_metrics(pool, gold, s)
            hits.append(in_topk)
            rank0.append(rank == 0)
            changed.append([d.doc_id for d in docs] != plain)
        rows[key(s)] = {
            "setting": s,
            "clean": {"gold_in_topk": st.mean(hits), "gold_rank0": st.mean(rank0),
                      "topk_changed": st.mean(changed)},
            "attacked": {},
        }

    for name in args.attacks:
        t0 = time.perf_counter()
        cfg = load_attack(name)
        attack = (build_attack(cfg, embedder_id=args.embedder, device=args.device)
                  if name == "C1" else build_attack(cfg))
        att_pools = []
        for q, gold, clean_pool in zip(queries, golds, clean_pools):
            # run_attack.py passes the clean top-k to the attack; the undefended
            # one is what the grid used.
            adv = attack.generate_adversarial_docs(q["query"], [c.doc for c in clean_pool[:TOP_K]])
            retriever.set_adversarial(adv)
            att_pools.append(record_pools(retriever, recorder, q["query"], gold))
            retriever.clear_adversarial()
        for s in settings:
            adv_n, hits = [], []
            for pool, gold in zip(att_pools, golds):
                docs, in_topk, _ = top_k_metrics(pool, gold, s)
                adv_n.append(sum(d.is_adversarial for d in docs))
                hits.append(in_topk)
            rows[key(s)]["attacked"][name] = {"adv_in_topk": st.mean(adv_n),
                                              "gold_in_topk": st.mean(hits)}
        logger.info(f"{name}: {time.perf_counter() - t0:.0f}s")
        del att_pools

    result = {"params": {"num_queries": len(queries), "offset": args.offset, "top_k": TOP_K,
                         "attacks": args.attacks,
                         "embedder": args.embedder},
              "rows": list(rows.values())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, args.out)
    plot(result, args.out)
    print(f"Wrote {args.out}, {args.out.with_suffix('.json')} and {args.out.with_suffix('.png')}")


def pct(x) -> str:
    return f"{100 * x:.1f}%"


def write_report(result: dict, out: Path) -> None:
    attacks = result["params"]["attacks"]
    L = [
        "# DoSRAGBench — Cluster filter sensitivity",
        "",
        f"*Generated by `scripts/sweep_cluster_filter.py`. Retrieval only (no LLM), "
        f"NQ, {result['params']['num_queries']} queries, top-{TOP_K}. Each setting is replayed "
        "on the same recorded candidate pools. `changed` = share of clean queries whose "
        "top-5 differs from the undefended top-5. Attack cells are adversarial docs in the "
        "top-5 / gold in the top-5.*",
        "",
        "| setting | clean gold@5 | clean gold@rank0 | clean changed | "
        + " | ".join(f"{a} adv@5 / gold@5" for a in attacks) + " |",
        "|---|---|---|---|" + "---|" * len(attacks),
    ]
    for r in result["rows"]:
        s = r["setting"]
        label = "**undefended**" if s is None else f"τ {s[0]:.2f}, ≥{s[1]} nbr, pool {s[2]}"
        if s is not None and abs(s[0] - 0.9) < 1e-9 and s[1] == 2 and s[2] == 50:
            label += " ← §3.15"
        c = r["clean"]
        cells = [f"{r['attacked'][a]['adv_in_topk']:.2f} / {pct(r['attacked'][a]['gold_in_topk'])}"
                 for a in attacks]
        L.append(f"| {label} | {pct(c['gold_in_topk'])} | {pct(c['gold_rank0'])} | "
                 f"{pct(c['topk_changed'])} | " + " | ".join(cells) + " |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


# Same tokens as plot_retrieval_gradient.py / plot_verdicts.py, so the figure
# reads as part of one set.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eda100", "#1baf7a"]  # one per min_neighbours value


def plot(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result["rows"] if r["setting"] is not None]
    pool = min(r["setting"][2] for r in rows)
    rows = [r for r in rows if r["setting"][2] == pool]
    undef = next(r for r in result["rows"] if r["setting"] is None)
    mns = sorted({r["setting"][1] for r in rows})
    attacks = result["params"]["attacks"]

    panels = [("Clean: gold in top-5", lambda r: 100 * r["clean"]["gold_in_topk"],
               100 * undef["clean"]["gold_in_topk"], "%")]
    panels += [(f"{a}: adversarial docs in top-5",
                (lambda a: lambda r: r["attacked"][a]["adv_in_topk"])(a),
                undef["attacked"][a]["adv_in_topk"], "docs") for a in attacks]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, f, base, unit) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.axhline(base, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
        for colour, mn in zip(SERIES, mns):
            pts = sorted((r["setting"][0], f(r)) for r in rows if r["setting"][1] == mn)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, linewidth=1.8,
                    marker="o", markersize=3.5, label=f"≥{mn} neighbour{'s' if mn > 1 else ''}",
                    zorder=3)
        ax.axvline(0.9, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("similarity threshold τ", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylabel(unit, fontsize=8.5, color=INK_SECONDARY)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        if unit == "docs":
            ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.text(0.01, 0.005, f"Dashed: undefended. Dotted: τ = 0.9 (§3.15). Pool = top-{pool}, "
             f"{result['params']['num_queries']} NQ queries, retrieval only.",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
