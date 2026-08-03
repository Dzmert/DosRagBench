"""Run a DoS attack experiment on a matched base/aligned model pair.

Two adversarial-injection modes:

  MERGE MODE (default, used for A-category and C1 displacement):
    The clean index is built once. Adversarial docs are held in an ephemeral
    secondary store and merged at query time. No per-query index rebuild, so
    this scales to 100k+ documents. Measures retrieval displacement (gold-doc
    eviction, adversarial pollution) and all generation-level metrics.

  INDEX-INJECTION MODE (--c1-latency, for the C1 algorithmic-complexity claim):
    Adversarial docs are added INTO the HNSW graph so they distort its
    structure and can trigger the O(log n) -> O(n) latency degradation that
    Category C targets. Because FAISS HNSW can't delete, this uses a single
    shared adversarial set across all queries (batch injection) rather than
    per-query rebuild, measuring aggregate retrieval-latency inflation.

Adversarial budget:
    --pollution-rate R sets the TOTAL adversarial budget as a fraction of the
    clean KB size (total_adv = int(R * len(kb_docs))). This is a realistic
    threat model: an attacker controls R of the corpus, not a fixed count per
    query. If --pollution-rate is omitted, falls back to the per-query count in
    the attack config (num_adversarial_docs).

Usage:
    python scripts/run_attack.py --category A1 --model-pair llama-3.1-8b --num-queries 200
    python scripts/run_attack.py --category C1 --model-pair llama-3.1-8b --c1-latency --pollution-rate 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dosragbench.attacks import BlockerAttack, BlockerConfig, build_attack
from dosragbench.metrics import QueryResult, classify_severity, compute_metrics
from dosragbench.models import detect_environment, load_model
from dosragbench.pipeline import Document, HNSWRetriever, RAGPipeline
from dosragbench.utils.config import (
    DATA_DIR,
    RESULTS_DIR,
    CACHE_DIR,
    load_attack,
    load_model_pair,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_dataset():
    queries_path = DATA_DIR / "queries.json"
    kb_path = DATA_DIR / "knowledge_base.json"
    if not queries_path.exists() or not kb_path.exists():
        raise FileNotFoundError("Missing data/. Run scripts/prepare_data.py first.")
    with open(queries_path) as f:
        queries = json.load(f)
    with open(kb_path) as f:
        raw_docs = json.load(f)
    docs = [Document(doc_id=d["doc_id"], text=d["text"]) for d in raw_docs]
    return queries, docs


def compute_kb_signature(kb_docs, embedder_id: str) -> str:
    """Cache key hashing the ordered doc-ids + embedder, so same-size subsets don't collide."""
    h = hashlib.md5()
    for d in kb_docs:
        h.update(d.doc_id.encode("utf-8"))
        h.update(b"\n")
    digest = h.hexdigest()[:12]
    return f"{len(kb_docs)}_{embedder_id.split('/')[-1]}_{digest}"


def get_or_build_index(retriever: HNSWRetriever, kb_docs, embedder_id, kb_signature):
    """Build the clean index, or load a cached one keyed by KB size + embedder."""
    cache_path = str(CACHE_DIR / f"index_{kb_signature}")
    if Path(f"{cache_path}.faiss").exists():
        logger.info(f"Loading cached index: {cache_path}.faiss")
        retriever.load_index(cache_path)
    else:
        retriever.build_index(kb_docs)
        try:
            retriever.save_index(cache_path)
        except Exception as exc:
            logger.warning(f"Could not cache index: {exc}")


def _pooled_adversarial_docs(attack, queries, num_queries, total_budget):
    """Generate a pooled adversarial set honoring a TOTAL budget.

    Distributes the budget across queries as evenly as possible: each query
    gets floor(total/n) docs, and the first (total mod n) queries get one extra.
    Falls back to the attack's own per-query count when total_budget is None.
    """
    q_slice = queries[:num_queries]
    all_adv = []

    if total_budget is None:
        for q in q_slice:
            all_adv.extend(attack.generate_adversarial_docs(q["query"], []))
        return all_adv

    n = len(q_slice)
    if n == 0:
        return all_adv
    base = total_budget // n
    remainder = total_budget % n

    # Pass the per-query allocation to attacks that accept a `count` (e.g. C1),
    # so a budget larger than the config's num_adversarial_docs is actually met.
    supports_count = "count" in inspect.signature(attack.generate_adversarial_docs).parameters

    for i, q in enumerate(q_slice):
        want = base + (1 if i < remainder else 0)
        if want <= 0:
            continue
        if supports_count:
            docs = attack.generate_adversarial_docs(q["query"], [], count=want)
        else:
            docs = attack.generate_adversarial_docs(q["query"], [])
        all_adv.extend(docs[:want])

    return all_adv


def make_blocker_feedback(rag, retriever):
    """Severity oracle for the BLOCKER baseline's black-box optimisation.

    Each call is one full RAG round-trip (retrieval + generation) against the
    live target, which is what makes this baseline's threat model stronger than
    ours: it needs query access to the deployed system. Returns the same
    SeverityLevel our own attacks are scored with, so the comparison is on one
    instrument. Latency is not part of the signal (lir=1.0) — the blocker
    targets refusal, and a single probe has no baseline latency to divide by.

    The returned callable carries a .calls counter so the cost is reportable.
    """

    def feedback(query: str, adv_docs: list[Document]) -> int:
        feedback.calls += 1
        retriever.set_adversarial(adv_docs)
        try:
            resp = rag.query(query)
        finally:
            retriever.clear_adversarial()
        return int(classify_severity(resp.answer))

    feedback.calls = 0
    return feedback


def run_model_side(model_config, retriever, queries, attack, top_k, num_queries,
                   prompt_style="auto"):
    """MERGE MODE: baseline + attacked for one model, no per-query rebuild."""
    logger.info(f"Loading model: {model_config.name}")
    loaded = load_model(model_config)
    rag = RAGPipeline(retriever=retriever, model=loaded, top_k=top_k,
                      prompt_style=prompt_style)

    # BLOCKER optimises against the live target, so its oracle can only be
    # bound once this side's model is loaded. Rebound per side: an adversarial
    # document optimised against the base model is not the same attack when
    # replayed against the aligned one.
    if isinstance(attack, BlockerAttack):
        attack.feedback_fn = make_blocker_feedback(rag, retriever)

    baseline_results, attacked_results = [], []

    for i, q in enumerate(queries[:num_queries]):
        query_text = q["query"]
        gold_doc_id = q.get("gold_doc_id")

        retriever.clear_adversarial()
        baseline_resp = rag.query(query_text, gold_doc_id=gold_doc_id)
        baseline_results.append(QueryResult.from_rag_response(baseline_resp))

        adv_docs = attack.generate_adversarial_docs(query_text, baseline_resp.retrieved_docs)
        retriever.set_adversarial(adv_docs)
        attacked_resp = rag.query(query_text, gold_doc_id=gold_doc_id)
        attacked_results.append(
            QueryResult.from_rag_response(attacked_resp, baseline_latency=baseline_resp.total_latency_s)
        )
        retriever.clear_adversarial()

        if (i + 1) % 25 == 0:
            msg = f"  [{i+1}/{num_queries}] processed"
            if isinstance(attack, BlockerAttack):
                msg += f" ({attack.feedback_fn.calls} oracle queries so far)"
            logger.info(msg)

    if isinstance(attack, BlockerAttack):
        logger.info(
            f"BLOCKER used {attack.feedback_fn.calls} oracle queries against "
            f"{model_config.name} ({attack.feedback_fn.calls / max(num_queries, 1):.1f} per query)"
        )

    loaded.unload()
    return {"baseline": baseline_results, "attacked": attacked_results}


def run_c1_latency(retriever, queries, attack, top_k, num_queries, embedder_id,
                   kb_docs, kb_signature, pollution_rate):
    """INDEX-INJECTION MODE: measure retrieval-latency degradation for C1."""
    logger.info("C1 latency mode: measuring retrieval degradation")

    # Clean baseline latencies
    retriever.clear_adversarial()
    clean_latencies, clean_gold_ranks = [], []
    for q in queries[:num_queries]:
        r = retriever.retrieve(q["query"], top_k=top_k, gold_doc_id=q.get("gold_doc_id"))
        clean_latencies.append(r.latency_s)
        clean_gold_ranks.append(r.gold_rank)

    # Determine total adversarial budget from pollution rate
    if pollution_rate is not None:
        total_budget = int(pollution_rate * len(kb_docs))
        logger.info(
            f"Pollution rate {pollution_rate:.3f} of {len(kb_docs)} KB docs "
            f"= {total_budget} adversarial docs total"
        )
    else:
        total_budget = None
        logger.info("No pollution rate set; using attack config per-query count")

    logger.info("Generating pooled adversarial documents...")
    all_adv = _pooled_adversarial_docs(attack, queries, num_queries, total_budget)
    logger.info(f"Injecting {len(all_adv)} adversarial docs into HNSW graph")

    polluted = HNSWRetriever(
        embedder_id=embedder_id,
        m=retriever.m,
        ef_construction=retriever.ef_construction,
        ef_search=retriever.ef_search,
        device=retriever.device,
    )
    polluted.build_index(kb_docs + all_adv)

    polluted_latencies, polluted_gold_ranks, adv_pollution = [], [], []
    for q in queries[:num_queries]:
        r = polluted.retrieve(q["query"], top_k=top_k, gold_doc_id=q.get("gold_doc_id"))
        polluted_latencies.append(r.latency_s)
        polluted_gold_ranks.append(r.gold_rank)
        adv_pollution.append(r.adversarial_in_topk)

    import statistics as st
    lir = [p / c for p, c in zip(polluted_latencies, clean_latencies) if c > 0]
    gold_evicted = sum(
        1 for cg, pg in zip(clean_gold_ranks, polluted_gold_ranks) if cg >= 0 and pg < 0
    )
    # Condition on queries whose gold was retrievable before the attack: a query
    # whose gold never made the clean top-k cannot be evicted from it, so counting
    # it in the denominator understates the attack. Same conditioning ASR already
    # applies via answerable-query filtering, and the definition compute_metrics
    # uses (metrics.py: gold_evicted / gold_present_baseline).
    gold_present_baseline = sum(1 for cg in clean_gold_ranks if cg >= 0)
    return {
        "mode": "c1_latency",
        "pollution_rate": pollution_rate,
        "kb_size": len(kb_docs),
        "num_queries": len(clean_latencies),
        "num_adversarial_docs": len(all_adv),
        "clean_retrieval_latency_ms_mean": round(st.mean(clean_latencies) * 1000, 3),
        "polluted_retrieval_latency_ms_mean": round(st.mean(polluted_latencies) * 1000, 3),
        "retrieval_lir_mean": round(st.mean(lir), 3),
        "retrieval_lir_median": round(st.median(lir), 3),
        "retrieval_lir_p90": round(sorted(lir)[int(len(lir) * 0.9)], 3),
        "gold_docs_evicted": gold_evicted,
        "gold_present_baseline": gold_present_baseline,
        "gold_eviction_rate": round(gold_evicted / gold_present_baseline, 3)
        if gold_present_baseline else 0.0,
        # Retained under its old name so previously-reported figures stay traceable.
        "gold_eviction_rate_unconditional": round(gold_evicted / len(clean_latencies), 3),
        "mean_adversarial_in_topk": round(st.mean(adv_pollution), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--model-pair", required=True)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto",
                        help="Embedder device: 'auto' (cuda if available), 'cuda', or 'cpu'. "
                             "Use cuda on Katana H200 for large-corpus index builds.")
    parser.add_argument("--side", choices=["both", "base", "aligned"], default="both")
    parser.add_argument("--c1-latency", action="store_true",
                        help="Run C1 retrieval-latency degradation mode (no generation)")
    parser.add_argument("--pollution-rate", type=float, default=None,
                        help="Total adversarial budget as a fraction of KB size "
                             "(e.g. 0.05 = 5%% of the corpus). Overrides per-query count.")
    parser.add_argument("--blocker-iters", type=int, default=25,
                        help="BLOCKER baseline: black-box search iterations per query. "
                             "The paper uses 1000; the default here is cut to keep a run "
                             "tractable (cost is iters x batch generations PER QUERY). "
                             "Pass 1000 to reproduce the paper exactly.")
    parser.add_argument("--blocker-batch", type=int, default=8,
                        help="BLOCKER baseline: candidates scored per iteration (paper: 32).")
    parser.add_argument("--blocker-tokens", type=int, default=50,
                        help="BLOCKER baseline: length of the generation half d_j (paper: 50).")
    parser.add_argument("--prompt-style", default="auto",
                        choices=["auto", "chat", "chat-no-refusal", "base"],
                        help="Force a RAG prompt regardless of the model's chat_template "
                             "flag. 'auto' reproduces every run recorded before "
                             "2026-08-03. Anything else is a prompt-confound ablation "
                             "and MUST be written to a separate --results-root.")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Where to write the run directory (default: results/). "
                             "Use results_promptablation/ for --prompt-style runs so the "
                             "62-run headline grid stays untouched — the collectors "
                             "scan results/ and results_hotpotqa/ only.")
    args = parser.parse_args()

    if args.prompt_style != "auto" and args.results_root is None:
        parser.error(
            "--prompt-style is an ablation and would overwrite the canonical run "
            f"directory results/{args.model_pair}_{args.category}. Pass an explicit "
            "--results-root (e.g. results_promptablation)."
        )

    if args.category == "BLOCKER" and args.c1_latency:
        parser.error(
            "--c1-latency is incompatible with BLOCKER: its optimisation needs a "
            "loaded generator for oracle feedback, and latency mode runs retrieval only."
        )

    logger.info(f"Environment: {detect_environment()}")
    pair = load_model_pair(args.model_pair)
    attack_config = load_attack(args.category)
    if args.num_queries is not None:
        attack_config.num_queries = args.num_queries

    queries, kb_docs = load_dataset()
    num_queries = min(attack_config.num_queries, len(queries))
    logger.info(f"Loaded {len(queries)} queries, {len(kb_docs)} KB docs; using {num_queries} queries")

    kb_signature = compute_kb_signature(kb_docs, args.embedder)

    retriever = HNSWRetriever(embedder_id=args.embedder, device=args.device)
    get_or_build_index(retriever, kb_docs, args.embedder, kb_signature)

    if args.category == "C1":
        attack = build_attack(attack_config, embedder_id=args.embedder, device=args.device)
    elif args.category == "BLOCKER":
        blocker_config = BlockerConfig(
            n_tokens=args.blocker_tokens,
            batch_size=args.blocker_batch,
            max_iters=args.blocker_iters,
            seed=attack_config.seed,
        )
        # feedback_fn is bound per model side in run_model_side(), once that
        # side's generator is loaded.
        attack = build_attack(attack_config, blocker_config=blocker_config)
        worst_case = args.blocker_iters * args.blocker_batch * num_queries
        logger.info(
            f"BLOCKER baseline: up to {worst_case:,} oracle generations per model side "
            f"({args.blocker_iters} iters x {args.blocker_batch} batch x {num_queries} queries). "
            f"Early aborts on refusal or a {blocker_config.stall_limit}-iteration stall."
        )
    else:
        attack = build_attack(attack_config)

    # Ablation runs carry the arm in the directory name, so several prompt styles
    # (and single-side arms) coexist under one root without overwriting each other.
    run_name = f"{args.model_pair}_{args.category}"
    if args.prompt_style != "auto":
        run_name += f"_prompt-{args.prompt_style}"
        if args.side != "both":
            run_name += f"_{args.side}"
    out_dir = (args.results_root or RESULTS_DIR) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {out_dir}/ (prompt_style={args.prompt_style})")

    if args.c1_latency:
        report = run_c1_latency(
            retriever, queries, attack, args.top_k, num_queries,
            args.embedder, kb_docs, kb_signature, args.pollution_rate,
        )
        # Filename encodes pollution rate so sweep runs don't clobber each other
        if args.pollution_rate is not None:
            fname = f"c1_latency_p{args.pollution_rate}.json"
        else:
            fname = "c1_latency.json"
        with open(out_dir / fname, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"C1 latency report: {json.dumps(report, indent=2)}")
        logger.info(f"Saved to {out_dir}/{fname}")
        return

    all_results, metrics_reports = {}, {}
    sides = []
    if args.side in ("both", "base"):
        sides.append(("base", pair.base))
    if args.side in ("both", "aligned"):
        sides.append(("aligned", pair.aligned))

    for side_name, model_cfg in sides:
        logger.info(f"══ {side_name.upper()}: {model_cfg.name} ══")
        t0 = time.perf_counter()
        results = run_model_side(model_cfg, retriever, queries, attack, args.top_k,
                                 num_queries, prompt_style=args.prompt_style)
        logger.info(f"{side_name} done in {time.perf_counter()-t0:.1f}s")

        all_results[side_name] = {
            "model_name": model_cfg.name,
            "hf_id": model_cfg.hf_id,
            "alignment_level": model_cfg.alignment_level,
            "baseline": [r.to_dict() for r in results["baseline"]],
            "attacked": [r.to_dict() for r in results["attacked"]],
        }
        report = compute_metrics(results["attacked"], results["baseline"], model_cfg.name, args.category)
        metrics_reports[side_name] = report.to_dict()
        logger.info(f"  ASR={report.asr:.3f} GDS={report.gds:.3f} "
                    f"LIR={report.lir_mean:.2f} retrieval_LIR={report.retrieval_lir_mean:.2f}")

    # Kept out of raw_results.json: the collectors iterate that file's top-level
    # keys as model sides, so an extra key there would be read as a third side.
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "model_pair": args.model_pair,
            "category": args.category,
            "side": args.side,
            "prompt_style": args.prompt_style,
            "num_queries": num_queries,
            "top_k": args.top_k,
            "embedder": args.embedder,
        }, f, indent=2)

    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_reports, f, indent=2)
    logger.info(f"Saved results to {out_dir}/")


if __name__ == "__main__":
    main()