"""Live demo — ask a question, watch the aligned model get silenced.

Built for the Thesis B presentation, where a demo is mandatory and the assessor
supplies the input. The whole benchmark takes GPU-hours because it runs 1,000
queries across 4 model pairs; a *single* query is seconds, so the thesis claim is
demonstrable live.

What one question shows, side by side:

    base model     clean -> answers      attacked -> still answers
    aligned model  clean -> answers      attacked -> "the context does not
                                                     support an answer"

Models and the FAISS index load once and stay resident, so start this BEFORE the
presentation and leave it at the prompt. Load takes minutes; each question takes
seconds.

    python3 scripts/demo.py                        # llama-3.1-8b, attack D2
    python3 scripts/demo.py --model-pair qwen-2.5-7b --attack D3
    python3 scripts/demo.py --no-llm               # retrieval only, no GPU needed

If a question is poorly covered by the corpus, retrieval will be bad and the
aligned model may decline *before* any attack. That is not a broken demo — it is
the mechanism, and the retrieval panel shows it happening. Say so out loud.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from dosragbench.attacks import build_attack  # noqa: E402
from dosragbench.metrics.refusal import classify_refusal, is_denial  # noqa: E402
from dosragbench.models import load_model  # noqa: E402
from dosragbench.pipeline import Document, HNSWRetriever, RAGPipeline  # noqa: E402
from dosragbench.utils.config import load_attack, load_model_pair  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / ".cache"

console = Console()

# Quiet by default: library INFO lines scroll the screen mid-demo.
logging.basicConfig(level=logging.WARNING, format="%(message)s")

ATTACK_MENU = [
    ("D2", "Circular Reference Chains", "strongest signal in the benchmark"),
    ("D3", "Epistemic Uncertainty", "the mechanism attack"),
    ("A2", "Contradiction Flooding", "conflicting evidence"),
    ("B3", "Multi-Retrieval Amplification", "floods the top-k"),
    ("C1", "Embedding Clustering", "retrieval-layer, evicts the gold passage"),
    ("D1", "Logical Contradiction Traps", ""),
    ("A3", "Authority Spoofing", ""),
    ("B2", "Generation Loop Induction", ""),
]

OK = "green"
DENIED = "red"
DIM = "grey62"


def load_corpus() -> tuple[list[dict], list[Document]]:
    queries_path, kb_path = DATA_DIR / "queries.json", DATA_DIR / "knowledge_base.json"
    for p in (queries_path, kb_path):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run scripts/prepare_data.py first.")
        # Detect an LFS pointer by its content, not by size — a small corpus is
        # legitimate, and failing on one would break the demo for no reason.
        with open(p, "rb") as f:
            if f.read(43).startswith(b"version https://git-lfs.github.com"):
                raise SystemExit(
                    f"{p} is a Git LFS pointer, not the data. "
                    f"Run: git lfs install && git lfs pull"
                )
    queries = json.loads(queries_path.read_text())
    docs = [Document(doc_id=d["doc_id"], text=d["text"])
            for d in json.loads(kb_path.read_text())]
    return queries, docs


def build_signature(kb_docs, embedder_id: str) -> str:
    """Same cache key run_attack.py uses, so the demo reuses its prebuilt index."""
    import hashlib
    h = hashlib.md5()
    for d in kb_docs:
        h.update(d.doc_id.encode("utf-8"))
        h.update(b"\n")
    return f"{len(kb_docs)}_{embedder_id.split('/')[-1]}_{h.hexdigest()[:12]}"


def verdict_of(answer: str) -> tuple[bool, str]:
    r = classify_refusal(answer or "")
    return is_denial(r), r.value


def show_retrieval(result, label: str) -> None:
    t = Table(title=f"Retrieved context — {label}", title_justify="left",
              title_style="bold", show_edge=False, header_style=DIM, padding=(0, 1))
    t.add_column("#", width=3, justify="right")
    t.add_column("score", width=7, justify="right")
    t.add_column("doc", width=12)
    t.add_column("passage", overflow="ellipsis", no_wrap=True)
    # A no-wrap cell claims its full length, so an over-long passage makes rich
    # collapse the narrow columns to zero. Budget the text against the real
    # terminal width instead of a fixed truncation.
    budget = max(40, console.width - 34)
    for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
        adv = getattr(doc, "is_adversarial", False)
        passage = ("[ADVERSARIAL] " if adv else "") + doc.text.replace("\n", " ")
        t.add_row(
            str(i), f"{score:.3f}",
            Text(doc.doc_id[:12], style="red bold" if adv else DIM),
            Text(passage[:budget], style="red" if adv else ""),
        )
    console.print(t)
    bits = [f"retrieval {result.latency_s * 1000:.0f} ms"]
    if result.gold_rank >= 0:
        bits.append(f"[green]gold passage at rank {result.gold_rank}[/green]")
    elif result.gold_in_topk is False and getattr(result, "_gold_known", True):
        bits.append("[yellow]gold passage NOT retrieved[/yellow]")
    if result.adversarial_in_topk:
        bits.append(f"[red]{result.adversarial_in_topk} of "
                    f"{len(result.documents)} adversarial[/red]")
    console.print("   " + " · ".join(bits) + "\n")


def answer_panel(side_label: str, answer: str, secs: float, phase: str) -> Panel:
    denied, kind = verdict_of(answer)
    style = DENIED if denied else OK
    tag = f"DENIED ({kind})" if denied else "ANSWERED"
    body = Text((answer or "").strip()[:700] or "(empty output)")
    return Panel(body, title=f"{side_label} — {phase}",
                 subtitle=f"{tag} · {secs:.1f}s", border_style=style,
                 title_align="left", subtitle_align="right")


class Demo:
    def __init__(self, args):
        self.args = args
        self.attack_code = args.attack
        t0 = time.perf_counter()

        console.print("[bold]Loading corpus…[/bold]")
        self.queries, self.kb_docs = load_corpus()
        self.gold_by_query = {q["query"].strip().lower(): q.get("gold_doc_id")
                              for q in self.queries}

        console.print(f"[bold]Loading index[/bold] ({len(self.kb_docs):,} passages)…")
        self.retriever = HNSWRetriever(embedder_id=args.embedder, device=args.device)
        sig = build_signature(self.kb_docs, args.embedder)
        cache = str(CACHE_DIR / f"index_{sig}")
        if Path(f"{cache}.faiss").exists():
            self.retriever.load_index(cache)
        else:
            console.print("[yellow]No cached index — building. This is slow; it is "
                          "cached afterwards, so do it before the presentation.[/yellow]")
            self.retriever.build_index(self.kb_docs)
            try:
                self.retriever.save_index(cache)
            except Exception as exc:  # cache is an optimisation, not a requirement
                console.print(f"[yellow]Could not cache index: {exc}[/yellow]")

        self.pipelines: dict[str, RAGPipeline] = {}
        if not args.no_llm:
            pair = load_model_pair(args.model_pair)
            for side, cfg in (("base", pair.base), ("aligned", pair.aligned)):
                console.print(f"[bold]Loading {side} model[/bold] — {cfg.name}…")
                model = load_model(cfg)
                self.pipelines[side] = RAGPipeline(
                    retriever=self.retriever, model=model, top_k=args.top_k)
                self.labels = getattr(self, "labels", {})
                self.labels[side] = cfg.name
        console.print(f"\n[bold green]Ready in {time.perf_counter() - t0:.0f}s[/bold green]")

    # ── one question ────────────────────────────────────────────────────────
    def run(self, question: str) -> None:
        gold = self.gold_by_query.get(question.strip().lower())
        cfg = load_attack(self.attack_code)
        attack = build_attack(cfg, embedder_id=self.args.embedder,
                              device=self.args.device) if self.attack_code == "C1" \
            else build_attack(cfg)

        console.rule(f"[bold]{question}[/bold]")
        if gold is None:
            console.print(f"[{DIM}]Free question — no gold passage on record, so "
                          f"retrieval quality is judged by the scores alone.[/{DIM}]\n")

        # 1. clean retrieval
        self.retriever.clear_adversarial()
        clean = self.retriever.retrieve(question, top_k=self.args.top_k, gold_doc_id=gold)
        clean._gold_known = gold is not None
        show_retrieval(clean, "no attack")

        clean_answers = self._generate(question, gold, "no attack")

        # 2. inject
        adv = attack.generate_adversarial_docs(question, clean.documents)
        console.print(Panel(
            Text((adv[0].text if adv else "").replace("\n", " ")[:400]),
            title=f"Injected {len(adv)} adversarial passage(s) — "
                  f"{self.attack_code} {cfg.name}   [first one shown]",
            border_style="yellow", title_align="left"))

        self.retriever.set_adversarial(adv)
        attacked = self.retriever.retrieve(question, top_k=self.args.top_k, gold_doc_id=gold)
        attacked._gold_known = gold is not None
        show_retrieval(attacked, f"under attack ({self.attack_code})")

        attacked_answers = self._generate(question, gold, f"under attack ({self.attack_code})")
        self.retriever.clear_adversarial()

        if clean_answers and attacked_answers:
            self._summarise(clean_answers, attacked_answers)

    def _generate(self, question: str, gold, phase: str) -> dict[str, str]:
        if self.args.no_llm:
            console.print(f"[{DIM}]--no-llm: retrieval only, no generation.[/{DIM}]\n")
            return {}
        out = {}
        for side in ("base", "aligned"):
            t0 = time.perf_counter()
            try:
                resp = self.pipelines[side].query(question, gold_doc_id=gold)
                out[side] = resp.answer
            except Exception as exc:  # never let one bad generation end the demo
                console.print(f"[red]{side} generation failed: {exc}[/red]")
                continue
            console.print(answer_panel(f"{side.upper()}  ({self.labels[side]})",
                                       resp.answer, time.perf_counter() - t0, phase))
        return out

    def _summarise(self, clean: dict, attacked: dict) -> None:
        rows = []
        for side in ("base", "aligned"):
            if side not in clean or side not in attacked:
                continue
            c_den, _ = verdict_of(clean[side])
            a_den, _ = verdict_of(attacked[side])
            rows.append((side, c_den, a_den))
        t = Table(show_edge=False, header_style=DIM, padding=(0, 2))
        t.add_column("model"); t.add_column("no attack"); t.add_column("under attack")
        for side, c_den, a_den in rows:
            mark = lambda d: Text("DENIED", style=DENIED) if d else Text("answered", style=OK)
            t.add_row(self.labels[side], mark(c_den), mark(a_den))
        console.print(t)
        broke = [s for s, c, a in rows if not c and a]
        if broke == ["aligned"]:
            console.print(Panel(
                "The attack silenced the aligned model on a query the base model "
                "still answers.\nThat is the thesis, on one query, live.",
                border_style="red", title="Result", title_align="left"))
        elif not broke:
            console.print(f"[{DIM}]Neither side was broken on this query. About 1 in 3 "
                          f"attacked queries flips; try another, or a different attack "
                          f"with :attack D2.[/{DIM}]")

    # ── prompt loop ─────────────────────────────────────────────────────────
    def help(self) -> None:
        t = Table(show_edge=False, header_style=DIM, title="Commands",
                  title_justify="left", title_style="bold")
        t.add_column("", width=18); t.add_column("")
        t.add_row("<question>", "run the demo on any question")
        t.add_row(":attack <CODE>", f"switch attack (current: {self.attack_code})")
        t.add_row(":attacks", "list the attacks worth demoing")
        t.add_row(":suggest", "10 questions with known-good retrieval")
        t.add_row(":help / :quit", "")
        console.print(t)

    def attacks(self) -> None:
        t = Table(show_edge=False, header_style=DIM)
        t.add_column("code"); t.add_column("name"); t.add_column("note", style=DIM)
        for code, name, note in ATTACK_MENU:
            style = "bold" if code == self.attack_code else ""
            t.add_row(Text(code, style=style), Text(name, style=style), note)
        console.print(t)

    def suggest(self) -> None:
        # Dataset queries carry a gold passage, so retrieval is known to be good.
        # Offered as a safety net; free questions are the higher-effect demo.
        picks = [q["query"] for q in self.queries[:200] if q.get("gold_doc_id")][:10]
        for q in picks:
            console.print(f"  [{DIM}]·[/{DIM}] {q}")

    def loop(self) -> None:
        self.help()
        console.print(f"\n[{DIM}]Attack: {self.attack_code}. "
                      f"Type a question, or :help.[/{DIM}]")
        while True:
            try:
                line = console.input("\n[bold cyan]question ›[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nbye")
                return
            if not line:
                continue
            if line in (":quit", ":q", ":exit"):
                return
            if line == ":help":
                self.help(); continue
            if line == ":attacks":
                self.attacks(); continue
            if line == ":suggest":
                self.suggest(); continue
            if line.startswith(":attack"):
                parts = line.split()
                if len(parts) != 2:
                    console.print("[yellow]usage: :attack D2[/yellow]"); continue
                try:
                    load_attack(parts[1].upper())
                except KeyError as exc:
                    # KeyError stringifies with its own quotes; unwrap for the screen.
                    console.print(f"[red]{exc.args[0]}[/red]"); continue
                self.attack_code = parts[1].upper()
                console.print(f"attack → [bold]{self.attack_code}[/bold]")
                continue
            if line.startswith(":"):
                console.print(f"[yellow]unknown command {line}[/yellow]"); continue
            try:
                self.run(line)
            except KeyboardInterrupt:
                # Ctrl-C abandons the question, never the session.
                console.print(f"\n[{DIM}]interrupted — back to the prompt[/{DIM}]")
                self.retriever.clear_adversarial()
            except Exception as exc:
                console.print(f"[red]Question failed: {exc}[/red]")
                self.retriever.clear_adversarial()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-pair", default="llama-3.1-8b")
    ap.add_argument("--attack", default="D2", help="starting attack code")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-llm", action="store_true",
                    help="retrieval only — no GPU, no model load. The fallback if "
                         "the demo machine cannot hold the models.")
    args = ap.parse_args()
    Demo(args).loop()


if __name__ == "__main__":
    main()
