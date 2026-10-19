"""Defended vs undefended: what each defence buys and what it costs.

Every run in results_defence/ is paired with the undefended grid run of the same
model pair and attack in results/. Both ran the same queries in the same order,
so the comparison is paired per query, and McNemar's exact test asks whether the
defence changed which queries were denied.

Per (run, side) it reports, undefended -> defended:
  benefit  attacked denial rate, attributable ASR, adversarial docs in top-k,
           gold-in-top-k under attack, attacked accuracy (NQ gold subset)
  cost     clean denial rate, clean gold-in-top-k and gold-at-rank-0, clean
           accuracy — a defence that blocks attacks by degrading clean answers
           has not mitigated anything

Denials are reclassified from the answer text by the validated classifier, and
accuracy uses the strict matcher, exactly as in compute_significance.py and
score_correctness.py.

Usage:
    python scripts/compare_defences.py
    python scripts/compare_defences.py --defence-dir results_defence --out results_defence/defence_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compute_significance import denied, mcnemar_exact_p, split_run  # noqa: E402
from score_correctness import outcome  # noqa: E402


def parse_arm(rest: str) -> tuple[str, str, str]:
    """'D2_def-cluster+rerank_aligned' -> ('D2', 'def-cluster+rerank', 'aligned')."""
    side = "both"
    for s in ("_aligned", "_base"):
        if rest.endswith(s):
            side, rest = s[1:], rest[: -len(s)]
    for marker in ("_def-", "_prompt-"):
        if marker in rest:
            attack, arm = rest.split(marker, 1)
            return attack, marker[1:] + arm, side
    return rest, "none", side


def side_metrics(side: dict, gold_by_query: dict) -> dict:
    B, A = side["baseline"], side["attacked"]
    n = len(B)
    clean_d = [denied(r) for r in B]
    att_d = [denied(r) for r in A]
    answerable = [i for i in range(n) if not clean_d[i]]
    gold = [i for i in range(n) if B[i]["query"] in gold_by_query]
    acc = lambda recs: (sum(outcome(recs[i]["answer"], gold_by_query[recs[i]["query"]]) == "C"  # noqa: E731
                            for i in gold) / len(gold)) if gold else None
    return {
        "n": n,
        "queries": [r["query"] for r in B],
        "clean_denied": clean_d,
        "att_denied": att_d,
        "clean_denial": sum(clean_d) / n,
        "att_denial": sum(att_d) / n,
        "asr": sum(att_d[i] for i in answerable) / len(answerable) if answerable else 0.0,
        "adv_in_topk": st.mean(r["retrieved_adversarial_count"] for r in A),
        "gold_clean": st.mean(1.0 if r["gold_in_topk"] else 0.0 for r in B),
        "gold_rank0_clean": st.mean(1.0 if r["gold_rank"] == 0 else 0.0 for r in B),
        "gold_att": st.mean(1.0 if r["gold_in_topk"] else 0.0 for r in A),
        "acc_clean": acc(B),
        "acc_att": acc(A),
    }


def paired_p(before: list[bool], after: list[bool]) -> float:
    b = sum(1 for x, y in zip(before, after) if x and not y)  # defence removed a denial
    c = sum(1 for x, y in zip(before, after) if not x and y)  # defence added a denial
    return mcnemar_exact_p(b, c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--defence-dir", type=Path, default=REPO_ROOT / "results_defence")
    ap.add_argument("--baseline-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <defence-dir>/defence_report.md")
    args = ap.parse_args()
    out = args.out or args.defence_dir / "defence_report.md"

    queries = json.loads((args.data_dir / "queries.json").read_text(encoding="utf-8"))
    gold_path = args.data_dir / "nq_gold_answers.json"
    gold_ids = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.exists() else {}
    gold_by_query = {q["query"]: gold_ids[q["query_id"]] for q in queries if q["query_id"] in gold_ids}

    rows, cache = [], {}
    for run_dir in sorted(p for p in args.defence_dir.iterdir() if p.is_dir()):
        raw_path = run_dir / "raw_results.json"
        if not raw_path.exists():
            continue
        family, rest = split_run(run_dir.name)
        attack, arm, _ = parse_arm(rest)
        base_dir = args.baseline_dir / f"{family}_{attack}"
        if not (base_dir / "raw_results.json").exists():
            print(f"  skip {run_dir.name}: no undefended run at {base_dir}")
            continue
        if base_dir not in cache:
            cache[base_dir] = json.loads((base_dir / "raw_results.json").read_text(encoding="utf-8"))
        undef = cache[base_dir]
        defended = json.loads(raw_path.read_text(encoding="utf-8"))
        for side in ("base", "aligned"):
            if side not in defended or side not in undef:
                continue
            u = side_metrics(undef[side], gold_by_query)
            d = side_metrics(defended[side], gold_by_query)
            if u["queries"][: d["n"]] != d["queries"]:
                print(f"  skip {run_dir.name}/{side}: query order differs from the undefended run")
                continue
            m = d["n"]
            rows.append({
                "run": run_dir.name, "family": family, "attack": attack, "arm": arm,
                "side": side, "n": m,
                "undefended": {k: v for k, v in u.items() if not isinstance(v, list)},
                "defended": {k: v for k, v in d.items() if not isinstance(v, list)},
                "p_att_denial": paired_p(u["att_denied"][:m], d["att_denied"]),
                "p_clean_denial": paired_p(u["clean_denied"][:m], d["clean_denied"]),
            })

    write_report(rows, out)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out} and {out.with_suffix('.json')} ({len(rows)} defended run-sides)")


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def arrow(u, d) -> str:
    return f"{pct(u)} → {pct(d)}"


def write_report(rows: list, out: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["family"], r["side"], r["attack"], r["arm"]))
    L = [
        "# DoSRAGBench — Defence experiment",
        "",
        "*Generated by `scripts/compare_defences.py`. Each defended run is paired per query "
        "with the undefended grid run (same pair, attack, queries). Values are undefended → "
        "defended. Denials reclassified by the validated classifier; accuracy is the strict "
        "NQ-open match over the gold subset. p = exact McNemar on which queries were denied.*",
        "",
        "## Benefit under attack",
        "",
        "| Pair | Side | Attack | Defence | Attacked denial | p | ASR | Adv docs in top-5 | Gold in top-5 (attacked) | Accuracy (attacked) |",
        "|---|---|---|---|---|---:|---|---|---|---|",
    ]
    for r in rows:
        u, d = r["undefended"], r["defended"]
        L.append(f"| {r['family']} | {r['side']} | {r['attack']} | {r['arm']} | "
                 f"{arrow(u['att_denial'], d['att_denial'])} | {r['p_att_denial']:.1e} | "
                 f"{arrow(u['asr'], d['asr'])} | {u['adv_in_topk']:.2f} → {d['adv_in_topk']:.2f} | "
                 f"{arrow(u['gold_att'], d['gold_att'])} | {arrow(u['acc_att'], d['acc_att'])} |")
    L += [
        "",
        "## Cost on clean queries",
        "",
        "Clean figures are the same for every attack within a pair, side and defence; one row each.",
        "",
        "| Pair | Side | Defence | Clean denial | p | Gold in top-5 | Gold at rank 0 | Accuracy (clean) |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    seen = set()
    for r in rows:
        key = (r["family"], r["side"], r["arm"])
        if key in seen:
            continue
        seen.add(key)
        u, d = r["undefended"], r["defended"]
        L.append(f"| {r['family']} | {r['side']} | {r['arm']} | "
                 f"{arrow(u['clean_denial'], d['clean_denial'])} | {r['p_clean_denial']:.1e} | "
                 f"{arrow(u['gold_clean'], d['gold_clean'])} | "
                 f"{arrow(u['gold_rank0_clean'], d['gold_rank0_clean'])} | "
                 f"{arrow(u['acc_clean'], d['acc_clean'])} |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
