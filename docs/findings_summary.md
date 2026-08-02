# DoSRAGBench — What the experiments actually used, found, and failed to find

*Companion to [`thesis_report_outline.md`](thesis_report_outline.md). Everything
here is traced to a file in this repo; nothing is inferred from documentation
alone. Where the documentation and the data disagree, the data wins and the
disagreement is flagged. Compiled 26 July 2026 against the 50-run kept set.*

---

## 1. The dataset

**BEIR NQ** — `BeIR/nq` corpus with `BeIR/nq-qrels` for gold labels. Not the
`nq_open` synthetic path, and not the offline fallback.

| Property | Value | Source |
|---|---|---|
| Corpus | BEIR Natural Questions (Wikipedia passages) | `dosrag_prep.log:1,26` |
| Full corpus size | 2,681,468 passages | `dosrag_prep.log:34` |
| Passages materialized | **501,000** (1,000 gold + 500,000 filler) | `dosrag_prep.log:35,37` |
| Queries | **1,000**, all with a qrels gold label | `dosrag_prep.log:36`, `data/queries.json` |
| Sampling seed | 42 | `prepare_data.py:254` |
| Gold definition | Highest-scoring qrels passage per query | `prepare_data.py:167-173` |
| Built | 6 Jul 2026, Katana job 8473011, 47 s walltime | `dosrag_prep.log:43-45` |

### Two things to know about this dataset

**`gold_answer` is empty for every query.** BEIR queries carry no short answer,
so `prepare_data.py:235` writes `""`. There is therefore **no answer-correctness
ground truth anywhere in the benchmark** — every result is derived from the
refusal/severity classifier deciding whether the model *declined*, never from
checking whether it was *right*. Defensible for a denial benchmark, but it must
be stated explicitly; readers will assume otherwise.

**The knowledge base is not in the working tree.** `data/knowledge_base.json` is
a 134-byte Git LFS pointer to a 276,668,938-byte object
(`.gitattributes:1`). Restore with `git lfs install && git lfs pull` — needs
`sudo apt install git-lfs` first on this machine.

### Documentation discrepancy — resolve before writing

`docs/positioning.md` ("BEIR NQ, ~500k passages") is **correct**.
`README.md` is **stale**: its quick-start describes a Natural Questions subset
with `--num-queries 50 --kb-size 1000` and a synthetic fallback, and its
"Expected Output" AVI table (A1 = 9.40, C1 = 1.17) is prototype placeholder
data, not the real results. Update or delete it — an examiner who reads the
README first will start from the wrong numbers.

---

## 2. Experimental configuration

### Retrieval
| Parameter | Value | Source |
|---|---|---|
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | `retriever.py:98` |
| Index | FAISS `IndexHNSWFlat`, inner-product metric | `retriever.py:132` |
| HNSW M | 16 | `retriever.py:99` |
| efConstruction | 200 | `retriever.py:100` |
| efSearch | 50 | `retriever.py:101` |
| top-k | 5 | `retriever.py:181` |

### Generation
| Parameter | Value | Source |
|---|---|---|
| Decoding | **Greedy, deterministic** (`do_sample=False`) | `loader.py:31` |
| max_new_tokens | 256 (2048 for R1-distill) | `configs/model_pairs.yaml` |
| Quantization | 4-bit, all 8 models | `configs/model_pairs.yaml` |
| Prompt | Two templates — see §5.2 | `rag.py:14,26,80-83` |

### Model pairs
| Pair | Base | Aligned | Alignment level |
|---|---|---|---|
| `llama-3.1-8b` | Llama 3.1 8B | Llama 3.1 8B Instruct | 0 → 2 |
| `qwen-2.5-7b` | Qwen 2.5 7B | Qwen 2.5 7B Instruct | 0 → 2 |
| `mistral-7b` | Mistral 7B v0.3 | Mistral 7B Instruct v0.3 | 0 → 1 |
| `llama-r1-8b` | Llama 3.1 8B Instruct | DeepSeek-R1-Distill-Llama-8B | 2 → 4 |

### Attack budget
Adversarial documents injected per query, from `configs/attacks.yaml`: 3 (A3),
5 (A1, B1, B2, D1, D3, D4), 6 (D2), 8 (A2), 10 (B3, C3), 200 (C1, C2, RAND).

⚠️ `configs/attacks.yaml` still declares `num_queries: 200` (or 50 for C2/C3),
but the real runs used **1,000**, overridden at the command line. The config
values are misleading as committed — fix them or the thesis methodology section
will contradict the config an examiner reads.

---

## 3. What worked

### 3.1 The headline
**12 genuine paradoxes across 3 of 4 model families**, passing both
FDR-corrected Fisher and within-model McNemar. Full breakdown and the
counting caveat are in `thesis_report_outline.md` §6.2 and
`results/verdict_breakdown.png`.

### 3.2 Which attacks do anything at all
Aggregated McNemar discordant counts across all runs — `c` = queries the attack
broke that worked clean, `b` = queries it "fixed". This measures attack potency
independent of the alignment question. Computed from `results/avi_significance.json`.

| Attack | c | b | | Attack | c | b |
|---|---:|---:|---|---|---:|---:|
| D2 Circular Reference Chains | 193 | 94 | | C1 Embedding Clustering | 97 | 27 |
| B3 Multi-Retrieval Amplification | 164 | 60 | | B2 Generation Loop Induction | 87 | 21 |
| A3 Authority Spoofing | 156 | 16 | | A2 Contradiction Flooding | 73 | 59 |
| D3 Epistemic Uncertainty | 143 | 99 | | A1 Guardrail Triggering | 60 | 37 |
| C2 Index Pollution | 107 | 75 | | D1 Logical Contradiction Traps | 54 | 47 |
| | | | | B1 Context Saturation | 28 | 11 |
| | | | | D4 Infinite Qualification | 11 | 13 |

**A3 has the cleanest signal in the entire benchmark** — 156 breaks against 16
repairs. It is also alignment-independent, which makes it the strongest control
condition: a demonstrably powerful attack that alignment neither worsens nor
mitigates.

### 3.3 The Qwen refusal floor — measured
Median clean-baseline denial rate across each family's runs
(`base_clean_floor` / `aligned_clean_floor` in `results/avi_report_clean.json`):

| Family | Base | Aligned |
|---|---|---|
| qwen-2.5-7b | 0.0% | **16.9%** |
| llama-3.1-8b | 0.4% | 0.3% |
| mistral-7b | 0.2% | 0.3% |
| llama-r1-8b | 0.3% | 0.0% |

Qwen-Instruct declines nearly one query in six *with no attack present*. This
single number is the entire justification for conditioning ASR on answerable
queries and for the McNemar test — put it in the thesis as a measured quantity,
not an aside.

### 3.4 C1 at the retrieval layer — full-scale curve
**Final.** All four points re-run at 501,000 passages, matching the main grid
(merged to `main` at 37ac2c8, 26 July 2026). Figure:
`results/c1_pollution_curve.png`; table: `results/c1_summary.csv`.

| Pollution | Adversarial docs | Gold eviction | Adversarial in top-5 | Retrieval LIR |
|---|---|---|---|---|
| 1% | 5,010 | 60.5% | 4.38 / 5 | 0.754 |
| 5% | 25,050 | 63.5% | 4.45 / 5 | 0.467 |
| 10% | 50,100 | **62.5%** | 4.47 / 5 | 0.433 |
| 20% | 100,200 | 65.5% | 4.53 / 5 | 0.449 |

**This overturns the earlier reading.** At the 87,925-passage scale the curve
appeared to climb steeply from 34% and saturate near 68%. At realistic scale
there is **no meaningful dose-response at all**: a **20× increase in attacker
budget** (5,010 → 100,200 documents) moves eviction by 5 percentage points. The
old "steep rise" was an artifact of the small corpus.

**The sequence is not even monotonic** — 60.5 → 63.5 → **62.5** → 65.5. The 10%
point sits *below* the 5% point, and that 1.0 pp dip is well inside the 1.5 pp
run-to-run difference measured in §5.4. Whatever upward drift the eye sees is
not distinguishable from noise on single runs. **Report this as flat at
60–65%**, not as a rising curve.

The correct framing is *efficiency, not dose-response*: **1% pollution already
achieves ~60% gold eviction**, and further investment is nearly wasted. That is
a stronger threat-model claim than a rising curve would have been — the attacker
needs far less corpus access than a dose-response story implies. Frame it that
way in the thesis.

Latency tells the same story in reverse: the ratio drops from 0.754 at 1% to
0.433 at 10% and then flattens, confirming §4.1 more sharply at scale. Note
retrieval is roughly **twice as fast** under attack at 10–20% pollution.

### 3.5 The clustering mechanism is confirmed by ablation
C1 versus RAND at identical budget: **62.0% versus 1.5% gold eviction**. See
§5.1 — this is the control that rules out corpus-size inflation, and it is
currently the most under-used result in the project.

---

## 4. What did not work

### 4.1 The latency attack failed outright
**The most important negative result in the project.** Across all 50 runs
(`results/avi_report_clean.json`):

| Metric | Min | Median | Max |
|---|---|---|---|
| Latency inflation, aligned | 0.958 | **1.082** | 1.771 |
| Latency inflation, base | 0.990 | 1.168 | 1.896 |
| Token overhead, aligned | 0.958 | **1.093** | 1.975 |
| Token overhead, base | 1.000 | 1.278 | 2.082 |

For C1 specifically, latency ratio is **below 1.0 at every pollution level** —
0.858, 0.688, 0.685, 0.608 (`results/c1_summary.csv`). Retrieval gets *faster*
as the attack intensifies: greedy HNSW descent converges more quickly once a
dense adversarial cluster sits near the query. Mechanism documented in
`docs/positioning.md:37-45`.

**Nothing in this benchmark is denial-of-service in the compute sense.** This is
the evidence base for reframing the thesis around induced denial / availability
(outline §7.4).

### 4.2 Attacks that produced nothing
- **D4 Infinite Qualification Traps** — 11 breaks against 13 repairs aggregated
  over all four pairs. Indistinguishable from noise everywhere. A designed
  attack that does not work.
- **B1 Context Window Saturation** — 28 breaks; significant only on Qwen, where
  the refusal floor does most of the work.

Report both as negative results. They cost nothing to state and they strengthen
the claim that the surviving effects are specific.

### 4.3 Dropped runs
Five runs excluded for fewer than 100 answerable queries on either side
(`scripts/clean_avi_report.py`, `--min-answerable 100`):

`llama-3.1-8b_A1_instructional` · `mistral-7b_A1` ·
`mistral-7b_A1_instructional` · `mistral-7b_C1` · `qwen-2.5-7b_A1_instructional`

**The whole `A1_instructional` arm was lost** — all three variants dropped, so
that condition has zero coverage. Either re-run it or remove it from the
methodology description.

### 4.4 Compute losses
70 jobs exited status 0; 8 failed (4 × exit 2, 4 × exit 1) with 9 tracebacks
across the root-level `DosRagBench.o*` / `DosRagC1.o*` logs. The identifiable
failure mode is a **401 Unauthorized on the gated `meta-llama/Llama-3.1-8B`
repository** — a HuggingFace access-token problem, not a compute failure.

---

## 5. Open issues (one resolved, two real)

### 5.1 ~~The RAND ablation has no results~~ — RESOLVED, it ran and it is strong
**Correction.** An earlier draft of this document claimed the RAND ablation
never produced output, because `results/llama-3.1-8b_RAND/` has no
`raw_results.json`. That was a false alarm: `--c1-latency` mode is
retrieval-only and never invokes the LLM, so it writes
`c1_latency_p<rate>.json` instead. The job ran successfully — Katana job
**8457869, exit status 0, 6m33s** (`ablation_katana.log`).

**The result is one of the strongest in the project.** At an identical budget
(500,200 passages, 25,010 injected documents, 200 queries, 5% pollution):

| | Gold eviction | Adversarial in top-5 | Retrieval latency ratio |
|---|---|---|---|
| **C1 (clustered)** | **62.0%** (124/200) | 4.41 / 5 | 0.457 |
| **RAND (random)** | **1.5%** (3/200) | 0.03 / 5 | 0.916 |

A **41× difference in eviction at the same document count.** Corpus-size
inflation is decisively ruled out as the mechanism — random injection at 25,010
documents barely perturbs retrieval at all. This belongs in the results chapter
as a headline control, not buried in an appendix, and `docs/positioning.md`
should cite these numbers directly instead of pointing at the script.

### 5.2 Base and aligned models receive different prompts
`rag.py:80-83` selects `RAG_PROMPT_CHAT` for models with a chat template and
`RAG_PROMPT_BASE` for those without. Chat-template availability is precisely
what distinguishes base from instruct, so **every AVI comparison confounds the
alignment effect with a prompt-wording effect.**

The templates are not merely formatting variants. The chat version
(`rag.py:14`) includes *"If the context doesn't contain the answer, say so
briefly"* — close to an instruction to refuse. The base version (`rag.py:26`)
does not.

This is the most attackable point in the methodology and it aims directly at the
dependent variable. Minimum: disclose it. Better: run the aligned model once
with the base prompt to bound the effect size.

### 5.3 ~~The C1 pollution curve mixes two corpus scales~~ — RESOLVED on `C1_Runs`
**Was:** three points at 87,925 passages and one at 500,200, because the
ablation job overwrote the 5% point at a six-times-larger corpus. Not a curve.

**Now:** the 1%, 10% and 20% points were re-run at full scale (branch
`C1_Runs`, commit 25f5a8e, logs `c1_p0.01.log` / `c1_p0.1.log` / `c1_p0.2.log`).
All four points are comparable — see §3.4 for the corrected table and the
reinterpretation it forces.

`results/c1_summary.csv` and `results/c1_pollution_curve.png` were regenerated
on 26 July 2026 and now reflect the full-scale data.

**A bug was found and fixed while regenerating them.** `scripts/collect_c1.py`
grouped results by `(kb_size, pollution_rate)` only, ignoring which attack
produced them. Once C1 and its RAND control both existed at ~500k passages and
5% pollution, the collector **averaged the attack together with its own
control**, reporting a meaningless 5% point (eviction 0.318 = mean of 0.62 and
0.015, `n_runs = 2`). The grouping key now includes the attack category,
recovered from the run directory name, and the plot draws C1 and RAND as
separate series. The bug did not affect earlier outputs, because before the
re-runs the two sat at different corpus sizes and grouped apart by accident.

The 5% point was re-run at full scale (branch `C1_Runs`, commit d083b22), so
all four points now sit at 501,000 passages.

### 5.4 An accidental replication — the only noise estimate in the C1 study
The 5% condition was measured **twice** under near-identical settings, because
the re-run superseded the ablation-era measurement at 500,200 passages:

| Run | KB size | Adversarial docs | Gold eviction | Retrieval LIR | Adv@k |
|---|---|---|---|---|---|
| Ablation (3 Jul) | 500,200 | 25,010 | 62.0% | 0.457 | 4.405 |
| Re-run (26 Jul) | 501,000 | 25,050 | **63.5%** | 0.467 | 4.445 |
| **Difference** | +0.16% | +40 | **+1.5 pp** | +0.010 | +0.040 |

**Keep this number.** It is currently the *only* evidence anywhere in the
project about C1 run-to-run variability, and the superseded 500,200 measurement
now survives only in git history (commit 25f5a8e).

It also sets the bar for the pollution trend. The entire apparent rise across
the sweep is **5 percentage points** (60.5% → 65.5%) for a **20× budget
increase** — only about three times the 1.5 pp difference between two runs of
the *same* condition. On one replication that is suggestive at best. State the
result as "eviction is approximately flat at 60–65% across the sweep" unless
you run proper seed repeats; claiming a rising dose-response is not supported.

---

## 6. Standing methodological weakness

The **refusal classifier is the measurement instrument for every number in this
thesis**, and it is 166 lines of pattern matching
(`src/dosragbench/metrics/refusal.py`, ~34 patterns) validated only by a
102-line unit-test file (`tests/test_refusal.py`). There is no hand-labelled
sample and no reported agreement rate.

Combined with §1's finding that `gold_answer` is empty — so nothing
cross-checks the classifier against actual answer quality — this is the single
highest-value gap to close before submission. Appendix C of the report outline
is scaffolded for it.
