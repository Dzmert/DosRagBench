"""Does the cluster filter remove the gold passage, and is the attack why?

§3.15.3 found that at tau = 0.83 / >=3 neighbours D2's attacked gold recall falls
from 72.7% to 60.7% while clean recall does not move, and inferred that the gold
passage is being swept up in the attack's cluster. The defended runs do not log
what the filter dropped, so this measures it directly, retrieval only, no LLM.

Per query, the top-`pool` candidates are recorded on the attacked query exactly
as sweep_cluster_filter.py does (same attack objects, seeds and query order).
For each filter setting it then asks, of the gold passage:

  in_pool        was it among the candidates the filter saw?
  removed        did the filter drop it?
  attack_caused  removed, but would have been kept had the adversarial
                 candidates not been counted as neighbours, i.e. it has fewer
                 than min_neighbours benign neighbours above tau
  lost_top5      it was in the undefended top-5 and is not in the defended top-5

Defended gold@5 is reported too; it should match the Katana runs' gold-in-top-5.

The same is reported on the clean pool, where attack_caused is zero by
construction; it is the false-removal baseline.

Usage:
    python scripts/trace_cluster_drops.py
    python scripts/trace_cluster_drops.py --attacks D2 --num-queries 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dosragbench.attacks import build_attack  # noqa: E402
from dosragbench.pipeline import HNSWRetriever  # noqa: E402
from dosragbench.pipeline.defence import ClusterFilter  # noqa: E402
from dosragbench.utils.config import DATA_DIR, load_attack  # noqa: E402
from run_attack import compute_kb_signature, get_or_build_index, gold_spec, load_dataset  # noqa: E402
from sweep_cluster_filter import TOP_K, PoolRecorder, record_pools  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUT_DIR = REPO_ROOT / "results_defence"
SETTINGS = [(0.90, 2), (0.83, 3)]  # the pre-registered setting and the tuned one


def trace(pool, gold_id, tau, mn, pool_k):
    """Classify what happened to the gold passage in one recorded pool."""
    pool = pool[:pool_k]
    ids = [c.doc.doc_id for c in pool]
    if gold_id not in ids:
        return {"in_pool": False, "removed": False, "attack_caused": False, "lost_top5": False,
                "in_top5": False}
    gi = ids.index(gold_id)
    kept = ClusterFilter(tau=tau, min_neighbours=mn).apply("", pool)
    kept_ids = [c.doc.doc_id for c in kept]
    removed = gold_id not in kept_ids
    emb = np.stack([c.embedding for c in pool])
    sims = emb @ emb[gi]
    adv = np.array([c.doc.is_adversarial for c in pool])
    near = sims > tau
    near[gi] = False
    benign_neighbours = int((near & ~adv).sum())
    return {
        "in_pool": True,
        "removed": removed,
        "attack_caused": removed and benign_neighbours < mn,
        "lost_top5": gi < TOP_K and gold_id not in kept_ids[:TOP_K],
        "in_top5": gold_id in kept_ids[:TOP_K],
        "adv_neighbours": int((near & adv).sum()),
    }


def summarise(traces):
    n = len(traces)
    in_pool = [t for t in traces if t["in_pool"]]
    removed = [t for t in in_pool if t["removed"]]
    return {
        "queries": n,
        "gold_in_pool": len(in_pool),
        "gold_removed": len(removed),
        "attack_caused": sum(t["attack_caused"] for t in removed),
        "lost_top5": sum(t["lost_top5"] for t in in_pool),
        "gold_top5": sum(t["in_top5"] for t in traces) / n,
        "median_adv_neighbours_when_removed": (
            float(np.median([t["adv_neighbours"] for t in removed])) if removed else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--attacks", nargs="+", default=["D2", "C1", "B3", "A2"])
    ap.add_argument("--num-queries", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "cluster_drops.md")
    args = ap.parse_args()

    queries, kb_docs = load_dataset(args.data_dir)
    queries = queries[: args.num_queries]
    retriever = HNSWRetriever(embedder_id=args.embedder, device=args.device)
    get_or_build_index(retriever, kb_docs, args.embedder, compute_kb_signature(kb_docs, args.embedder))
    recorder = PoolRecorder()
    retriever.defences = [recorder]
    retriever.defence_pool_k = args.pool

    retriever.clear_adversarial()
    golds = [gold_spec(q) for q in queries]
    clean_pools = [record_pools(retriever, recorder, q["query"], g) for q, g in zip(queries, golds)]

    rows = []
    for tau, mn in SETTINGS:
        rows.append({"condition": "clean", "tau": tau, "mn": mn,
                     **summarise([trace(p, g, tau, mn, args.pool) for p, g in zip(clean_pools, golds)])})

    for name in args.attacks:
        t0 = time.perf_counter()
        cfg = load_attack(name)
        attack = (build_attack(cfg, embedder_id=args.embedder, device=args.device)
                  if name == "C1" else build_attack(cfg))
        att_pools = []
        for q, gold, clean_pool in zip(queries, golds, clean_pools):
            adv = attack.generate_adversarial_docs(q["query"], [c.doc for c in clean_pool[:TOP_K]])
            retriever.set_adversarial(adv)
            att_pools.append(record_pools(retriever, recorder, q["query"], gold))
            retriever.clear_adversarial()
        for tau, mn in SETTINGS:
            rows.append({"condition": name, "tau": tau, "mn": mn,
                         **summarise([trace(p, g, tau, mn, args.pool) for p, g in zip(att_pools, golds)])})
        logger.info(f"{name}: {time.perf_counter() - t0:.0f}s")

    result = {"params": {"num_queries": len(queries), "pool": args.pool, "top_k": TOP_K,
                         "embedder": args.embedder}, "rows": rows}
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, args.out)
    print(f"Wrote {args.out} and {args.out.with_suffix('.json')}")


def write_report(result: dict, out: Path) -> None:
    p = result["params"]
    L = [
        "# DoSRAGBench — What the cluster filter removes",
        "",
        f"*Generated by `scripts/trace_cluster_drops.py`. Retrieval only, NQ, {p['num_queries']} "
        f"queries, pool {p['pool']}, top-{p['top_k']}. `removed` = gold was in the pool and the "
        "filter dropped it. `attack-caused` = removed, but it has fewer than min-neighbours benign "
        "neighbours above τ, so the adversarial candidates tipped it over. `lost top-5` = gold was "
        "in the undefended top-5 and is not in the defended top-5.*",
        "",
        "| Condition | Setting | Gold in pool | Gold removed | Attack-caused | Lost from top-5 | "
        "Median adv. neighbours of removed gold | Defended gold@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["rows"]:
        med = r["median_adv_neighbours_when_removed"]
        L.append(f"| {r['condition']} | τ {r['tau']:.2f} / ≥{r['mn']} | {r['gold_in_pool']} | "
                 f"{r['gold_removed']} | {r['attack_caused']} | {r['lost_top5']} | "
                 f"{'—' if med is None else f'{med:g}'} | {100 * r['gold_top5']:.1f}% |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
