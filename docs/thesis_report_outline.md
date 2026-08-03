# DoSRAGBench — Thesis Report Skeleton

*A section-by-section template for the written thesis. Each section states what it
must accomplish, roughly how long it should be, and which artifact in this repo
feeds it. Word budgets assume a ~15,000-word research thesis; scale
proportionally if your course specifies otherwise.*

*Companion document: [`findings_summary.md`](findings_summary.md) — the audited
record of what the experiments actually used, found, and failed to find, with
file citations. Read it before writing Chapters 4–7. It flags two issues to
resolve first (a prompt confound in the AVI comparison, and a mixed-scale C1
pollution curve whose summary CSV is stale), and one strong, under-used result:
the C1-vs-RAND ablation at 62.0% versus 1.5% gold eviction.*

---

## The spine (write this down before writing anything else)

Everything in the report should serve one of these five sentences. If a paragraph
doesn't, cut it.

1. RAG systems inherit an availability risk that the security literature has not
   measured, because prior work studies *misdirection* (make the model say the
   wrong thing) rather than *denial* (make the model unable to say anything).
2. Instruction tuning plausibly makes this worse, because deferring to the
   retrieved context is a trained behaviour an attacker can invoke by making that
   context look inadequate.
3. To test that, you need a matched base-vs-aligned design, a graded denial
   metric, a *validated* refusal classifier, and — critically — a filter that
   separates attack-induced denial from the aligned model's baseline refusal
   habit.
4. Across 13 attacks × 4 model pairs × 2 datasets, 51 of 62 runs point in the
   paradox direction and 39 survive that filter.
5. The mechanism is **context-faithfulness, not safety**. Genuine safety refusals
   are 0.02% of aligned responses; epistemic refusals are 30% (NQ) to 80%
   (HotpotQA). Three independent lines converge — composition, a monotonic
   dose-response in retrieval quality, and a reasoning-distilled pair that
   reverses direction. See `docs/reframing.md`.
6. So the claim is sharper than the original hypothesis: alignment creates a
   *specific* vulnerability to apparent evidential inadequacy, and the method for
   telling real paradoxes from floor artifacts is
   itself a contribution.

**Framing decision to settle before you write Chapter 1:** the slides
(`docs/thesis_meeting_slides.md`, Slide 8) flag that "DoS" oversells what you
measured — latency barely moves (LIR/TOR ≈ 1.0–1.3×), and the denial is
semantic. Either retitle around *induced denial / availability degradation* and
keep DoS as the motivating frame, or keep the name and spend a section defending
it. Do not leave this ambiguous; a reader who expects compute exhaustion and
finds refusal rates will distrust the whole chapter.

---

## Front matter

- **Title page, declaration of originality, acknowledgements** — per your course
  template, not this file.
- **Abstract** (250–300 words). Write it *last*. Structure: one sentence on RAG
  availability as an unstudied risk → one on the hypothesis → two on the design
  (13 attacks, 4 matched pairs, ~1000 NQ queries, AVI) → two on the headline
  (27 raw → 12 genuine; epistemic yes, retrieval no) → one on the implication.
  Put a real number in it. Abstracts without numbers read as proposals.
- **Contents, list of figures, list of tables, nomenclature.** Auto-generate.
  Define AVI, ASR, GDS, LIR, TOR, CDR in the nomenclature list *and* again at
  first use in the text — examiners skim.

---

## 1. Introduction — ~1,200 words

Purpose: make the examiner want to read Chapter 4, and tell them exactly what
you claim.

- **1.1 Motivation.** RAG is deployed in production (support bots, enterprise
  search, legal/medical retrieval). Availability is a security property. An
  attacker who can insert documents into a corpus — a wiki, a shared drive, a
  crawled site — can attack availability without ever touching the model.
- **1.2 The alignment paradox hypothesis.** Safety tuning trains a model to
  refuse. Refusal is a *capability the attacker can invoke*. State it as a
  falsifiable claim, not a slogan.
- **1.3 Research questions.** Three, numbered, referenced again in the
  conclusion:
  - RQ1: Does instruction tuning increase susceptibility to attack-induced
    denial?
  - RQ2: *Which trained behaviour* drives it — safety guardrails or
    context-faithfulness? This is the question the data answers decisively and
    against the original hypothesis; it should carry the discussion chapter.
  - RQ3: How much of the apparent effect is an artifact of the aligned model's
    baseline refusal floor?
  RQ3 is your differentiator — it is the question that turns 27 into 12. Give it
  equal billing, not a footnote.
- **1.4 Contributions.** Bulleted, four or five, each one sentence:
  benchmark + 13 attacks; AVI and the attack-attributable ASR definition; the
  Fisher + McNemar filtering protocol; the empirical finding
  (epistemic-specific, cross-family); the negative results (A3
  alignment-independent, R1-distill protective) as controls.
- **1.5 Thesis structure.** One short paragraph. Do not pad this.

---

## 2. Background and related work — ~2,500 words

Purpose: establish that nobody has measured this, and that your threat model is
weaker (therefore stronger as a result) than prior work's.

- **2.1 Retrieval-augmented generation.** Dense retrieval, embedding models,
  HNSW/ANN indexes. Keep it tight — enough to justify why HNSW is the realistic
  target (Pinecone, Weaviate, Milvus, Qdrant all use it).
- **2.2 Alignment and groundedness.** RLHF/DPO and instruction tuning, then split
  refusal into its two trained sources: *safety* refusal (policy-driven) and
  *context-faithfulness* / groundedness training ("answer only from the retrieved
  passages", the standard defence against RAG hallucination). Cite the
  groundedness and faithfulness literature here, not just the safety literature —
  it is the mechanism the results identify. Framing refusal as a learned response
  to context sets up RQ2, which asks which of the two an attacker can invoke.
- **2.3 Attacks on RAG.** Corpus poisoning, prompt injection, jailbreaks. This is
  where `docs/positioning.md` goes almost verbatim — the PoisonedRAG /
  CorruptRAG / Zhong et al. comparison table, and the three differentiators
  (denial not misdirection; grey-box optimization-free; benchmark not single
  attack). **Reuse that table as Table 2.1.**
- **2.4 Denial of service on ML systems.** Sponge examples, Indyk–Xu (NeurIPS
  2023) worst-case ANN complexity. This is also where you honestly place your
  scope note: the O(n) traversal attack is future work, and the C-family reduces
  latency rather than increasing it.
- **2.5 Gap statement.** One paragraph, explicit: prior work measures whether a
  poisoned passage steers an answer; none measure whether the system is pushed
  into *not answering*, and none contrast base against aligned models on it.

---

## 3. Threat model and metrics — ~1,800 words

Purpose: make the evaluation defensible before any results appear. Separating
this from Chapter 4 is worth it — examiners attack definitions, and you want
them isolated and airtight.

- **3.1 Attacker capabilities.** Can insert documents into the corpus; knows the
  embedder (grey-box, realistic for open embedders); no gradient access, no
  model weights, no query-time control. State what the attacker *cannot* do —
  that's what makes the threat model credible.
- **3.2 Defining denial.** The graded severity scale. Distinguish safety
  refusal, epistemic refusal, hedged non-answer, generation failure — this is
  what `metrics/refusal.py` implements and what the 18 unit tests cover. Include
  a worked example of each class from real transcripts.
- **3.3 The six metrics.** ASR, GDS, LIR, TOR, CDR — formal definition, units,
  range, and what each is sensitive to. Be explicit that LIR/TOR are latency and
  token measures that turn out to be near-null, and that this is a *finding*, not
  a gap.
- **3.4 Attack-attributable ASR.** The conditional definition: of queries the
  model answers with no attack, the fraction the attack pushes into full denial.
  Explain in words *why* the naive unconditional version is wrong — aligned Qwen
  refuses ~17% of queries with no attack at all, and naive ASR credits the
  attack for those.
- **3.5 AVI.** Definition, the ε = 0.01 floor on base ASR, and the honest caveat:
  when base ASR ≈ 0, a large AVI means "the attack only works on the aligned
  model," not that the ratio is precise. Say this here, once, properly, so you
  can point back to it every time a 21.18 appears in a table.

---

## 4. Benchmark design and implementation — ~2,500 words

Purpose: reproducibility. An examiner should be able to rebuild this.

- **4.1 Architecture.** One figure: corpus → embedder → HNSW index → top-k →
  prompt → generator → refusal classifier → metrics. Map each box to a module
  (`pipeline/retriever.py`, `pipeline/rag.py`, `metrics/refusal.py`). The README
  project tree is the raw material; redraw it as a diagram, don't paste the tree.
- **4.2 Attack taxonomy.** Four families, 13 attacks. One subsection per family
  (A guardrail/authority, B context/generation, C retrieval/embedding, D
  epistemic/logical), each with: mechanism, why it should cause denial, and one
  concrete adversarial passage as a figure. **The example passages matter more
  than the prose** — they are the most memorable thing in the chapter.
  Source: `src/dosragbench/attacks/*.py`, `configs/attacks.yaml`.
- **4.3 Model pairs.** The four pairs and why each is chosen. Emphasise the
  matched-pair design as what makes AVI a clean contrast, and flag the
  R1-distill pair as the "same architecture, different post-training" arm.
  Source: `configs/model_pairs.yaml`.
- **4.4 Experimental protocol.** Exact figures are tabulated in
  `findings_summary.md` §1–2: BEIR NQ, 1,000 queries, 501,000 passages
  (1,000 gold + 500,000 filler, seed 42) drawn from a 2,681,468-passage corpus;
  `all-MiniLM-L6-v2`; FAISS HNSW M=16 / efConstruction=200 / efSearch=50;
  top-k=5; greedy decoding at 256 new tokens; 4-bit quantization; Katana HPC.
  Also state the 56 runs → 50 kept and the ≥100-answerable-query exclusion rule
  *before* the results, so it reads as protocol rather than post-hoc filtering,
  and name the five dropped runs (the whole `A1_instructional` arm is among them).
  Note explicitly that BEIR carries no short answers, so `gold_answer` is empty
  and denial is measured by the classifier, never by answer correctness.
- **4.5 Controls and baselines.** The RAND random-document baseline and what it
  isolates. At identical budget (500,200 passages, 25,010 injected documents),
  C1 evicts the gold passage for **62.0%** of queries while RAND evicts **1.5%**
  — a 41× separation that rules out corpus-size inflation as the mechanism.
  Report these numbers directly (`results/llama-3.1-8b_{C1,RAND}/c1_latency_p0.05.json`),
  not as a pointer to the ablation script.

---

## 5. Statistical methodology — ~1,200 words

Purpose: this is your methodological contribution. Give it its own chapter so it
can be cited as one.

- **5.1 Why raw AVI is insufficient.** Two failure modes: the baseline refusal
  floor, and the near-zero denominator.
- **5.2 Wilson confidence intervals** on every ASR. Why Wilson and not normal
  approximation (proportions near 0).
- **5.3 Fisher's exact test** — the cross-model question: do aligned and base
  deny different fractions of answerable queries?
- **5.4 McNemar's test** — the within-model question: does the *attack* break
  queries that worked clean (c vs b), or is the model just refusal-prone?
- **5.5 The conjunction rule.** A run counts as a genuine paradox only if it
  passes both. State the four verdict classes here — genuine, floor artifact,
  attack-but-independent, protective — and define them precisely. Everything in
  Chapter 6 refers back to these labels.
- **5.6 Multiple comparisons.** Benjamini–Hochberg FDR across all 50 tests,
  α = 0.05.

Source: `scripts/compute_significance.py`, `results/avi_significance.md`.

---

## 6. Results — ~3,000 words

Purpose: report, don't interpret. Interpretation is Chapter 7. This separation
is what keeps a results chapter honest.

- **6.1 Overview.** The 50-run grid. Figure: `results/avi_summary.png` (two-panel
  heatmap). **Caveat from the slides — it plots raw AVI, so annotate the genuine
  cells or pair it immediately with the 6.2 breakdown, or it oversells.**
- **6.2 Raw vs. filtered.** Figure: `results/verdict_breakdown.png`.
  27/50 runs point in the paradox direction; of those 27, the conjunction rule
  keeps **12 genuine** and rejects **5 floor artifacts, 2
  attack-but-alignment-independent, and 8 not-significant**. The remaining 23
  runs point the other way (base denies more) and split **8 protective, 15 not
  significant**. Verdicts partition all 50 runs.
  **Do not write "breakdown of the 27: 12/5/2/8 protective."** Protective is
  defined by the base model denying *more*, so it cannot be inside the
  paradox-direction group. The two groups happen to contain 8 runs each, which
  is exactly why the mistake is easy to make and hard to spot.
  This is the single most important paragraph in the thesis — give it a table
  and a sentence a reader can quote.
- **6.3 The genuine paradoxes.** Full table from `results/avi_significance.md`
  (or the GENUINE subset, with the full table in an appendix). Lead with the
  cleanest case: **D3 epistemic uncertainty on Llama-3.1, base 0.8% [0.4–1.6] →
  aligned 8.4% [6.9–10.3], AVI 8.43, McNemar 84/3, p = 1.4 × 10⁻²¹**.
  Non-overlapping CIs and an overwhelming paired effect — walk the reader
  through why this one is unarguable.
- **6.4 Pattern by attack family.** Genuine paradoxes: Qwen-2.5 (**8** — A1, A3,
  B1, B2, B3, C1, C2, D2), Llama-3.1 (3 — B2, D1, D3), Mistral (1 — D2).
  Every non-Qwen genuine paradox is a reasoning/epistemic attack; none are the
  retrieval C-attacks. Cross-family presence (3 of 4 model families) is what
  rules out a single-model quirk.
- **6.5 Floor artifacts.** The five demoted Qwen runs (A2, C3, D1, D3, D4) with
  their large raw AVIs alongside their non-significant McNemar. Present this
  as a result, not an apology — demoting your own biggest numbers is the
  strongest rigor signal in the document.
- **6.6 Controls.** A3 Authority Spoofing — a real attack (McNemar 48/0, 60/0)
  that is alignment-*independent* (Fisher n.s.). R1-distill — protective almost
  everywhere. These bracket the effect: it is specific, not a pipeline artifact.
- **6.7 Latency and token cost.** The near-null LIR/TOR/CDR results from
  `results/avi_report.md`. Report plainly: the attacks do not meaningfully
  increase compute. This is the empirical basis for the reframing in 7.4.
- **6.8 C1 retrieval-level results.** Pollution curve
  (`results/c1_pollution_curve.png`), gold-passage eviction, and the C1-vs-RAND
  ablation. Note that eviction happens at the retrieval layer without
  necessarily producing generation-level denial — that gap is itself
  interesting.

---

## 7. Discussion — ~2,500 words

Purpose: the "so what". This chapter is what separates a distinction from a
pass.

- **7.1 Answering RQ1–RQ3.** One subsection each, explicitly numbered back to
  Chapter 1. RQ1: qualified yes. RQ2: epistemic, not retrieval. RQ3: substantial
  — 15 of 27 apparent paradoxes fail one of the two tests.
- **7.2 Why epistemic attacks?** Your mechanistic hypothesis: alignment tuning
  rewards expressing calibrated uncertainty and declining under conflicting
  evidence, so adversarial context that manufactures uncertainty or contradiction
  invokes a *trained* behaviour. Retrieval attacks degrade evidence without
  invoking it. Be explicit that this is a hypothesis your data is consistent
  with, not something you demonstrated causally — and say what experiment would
  test it (e.g. probing refusal-head activations, or DPO-stage ablations).
- **7.3 Why is Qwen worst?** 7 of 12 genuine paradoxes. Candidate explanations:
  heavier refusal tuning, different RLHF data, the ~17% baseline floor
  interacting with attack pressure. Offer the alternatives and say which the
  data favours.
- **7.4 The DoS framing.** Argue the reframe from the 6.7 evidence: what you
  measured is *induced denial / availability degradation*, and compute-DoS on
  RAG remains open. Owning this is far stronger than letting an examiner find it.
- **7.5 Defensive implications.** What a practitioner does with this: corpus
  provenance, retrieval-time anomaly detection, refusal-rate monitoring as an
  availability SLO. Be concrete; a paragraph of vague "future defences" is worse
  than three specific mechanisms.
- **7.6 Limitations.** Single embedder (all-MiniLM-L6-v2); template-based rather
  than gradient-optimized adversarial text; pattern-based refusal classifier
  (state its validation and its failure modes); no defence evaluation; NQ-only
  domain; 7–8B models only. Every one of these already appears in the README's
  known-limitations section — expand each into two or three honest sentences.
  **Add the two that the README does not list**, both from `findings_summary.md`
  §5–6: (i) base and aligned models receive *different prompt templates*
  (`rag.py:80-83`), and the chat template contains language close to an
  instruction to refuse — so every AVI comparison confounds alignment with
  prompt wording; (ii) with `gold_answer` empty there is no answer-correctness
  check anywhere, so the refusal classifier is an unvalidated single point of
  measurement. Disclosing both is far stronger than having an examiner find them.

---

## 8. Conclusion and future work — ~800 words

- **8.1 Summary of contributions.** Mirror 1.4, now in past tense with numbers.
- **8.2 Future work.** Prioritised, not a list of everything imaginable:
  (i) mechanism — probing *why* epistemic attacks land; (ii) genuine
  compute-DoS via traversal-lengthening (Indyk–Xu O(n)) with a jitter-robust
  timing methodology; (iii) defence evaluation — perplexity filtering, NLI-based
  contradiction detection; (iv) breadth — more embedders, larger models,
  non-NQ domains.
- **8.3 Closing.** One paragraph. The reframed thesis in a single sentence.

---

## References

Use a manager (Zotero/BibTeX) from the first citation, not retrofitted.
Anchor set: PoisonedRAG (Zou et al., 2024), CorruptRAG (2025), Zhong et al.
(EMNLP 2023) corpus poisoning, Indyk–Xu (NeurIPS 2023), MutedRAG, the RAG
foundation papers, and the alignment/RLHF references behind Chapter 2.2.

---

## Appendices

- **A. Full 50-run significance table** — the complete
  `results/avi_significance.md`, so Chapter 6 can show the GENUINE subset only.
- **B. Attack passage examples** — one full adversarial document per attack, all
  13. Cheap to produce, disproportionately convincing.
- **C. Refusal classifier** — pattern rules, the 18 unit tests, and a
  hand-labelled validation sample with agreement rate. *If you have not done the
  hand-labelled validation, do it before submission* — the classifier is the
  measurement instrument for every number in the thesis, and it is the single
  most attackable point in the methodology.
- **D. Reproduction instructions** — the README quick-start, plus exact commands,
  configs, and the Katana job scripts.
- **E. Excluded runs** — the 5 dropped runs and why.

---

## Suggested writing order

Not front to back. Write in this order so that the hardest thinking happens
while you still have time to run an experiment if a hole appears:

1. **Chapter 6 (Results)** — you already have every number; this is transcription.
2. **Chapter 5 (Statistics)** — the reasoning is already in
   `compute_significance.py` and the "how to read this" section of the report.
3. **Chapter 3 (Threat model and metrics)** — definitions, written once, carefully.
4. **Chapter 4 (Design)** — mostly assembling what exists in the repo.
5. **Chapter 7 (Discussion)** — the chapter that needs the most uninterrupted thought.
6. **Chapter 2 (Background)** — expand from `docs/positioning.md`; the literature
   sweep is the most interruptible task, so it fits around everything else.
7. **Chapters 1 and 8, then the abstract** — write the framing once you know
   exactly what you're framing.

## Figures worth making early

Figures take longer than expected and drive the prose, so build them before
writing the chapter they sit in.

- Pipeline architecture diagram (§4.1) — does not exist yet.
- Attack taxonomy tree, 4 families × 13 attacks (§4.2) — does not exist yet.
- Annotated version of `avi_summary.png` marking the 12 genuine cells (§6.1).
- ~~Verdict breakdown~~ — **done**: `results/verdict_breakdown.png`, from
  `scripts/plot_verdicts.py`.
- ~~Forest plot of the 12 genuine runs~~ — **done**:
  `results/forest_genuine.png`, from `scripts/plot_forest.py`.

Both read `results/avi_significance.json`, which `scripts/compute_significance.py`
now emits alongside the markdown table, so the figures and the tables cannot
drift apart. Regenerate with:

```bash
python scripts/compute_significance.py
python scripts/plot_forest.py
python scripts/plot_verdicts.py
```
