# Positioning C1 against prior RAG corpus-poisoning work

The C1 attack shares its *surface* with existing RAG poisoning attacks — it
injects adversarial passages that get retrieved — but differs in **objective**
and **construction**. This is the framing to use in the related-work section so
reviewers do not read C1 as a re-run of PoisonedRAG.

**Scope of this document:** it defends **C1 only**. The thesis-level positioning —
the denial/jamming neighbours, the context-faithfulness literature behind the
mechanism claim, and the ranked list of differentiators — is argued elsewhere.
The strongest novelty claim is the matched base→aligned design, not any single
attack, and it is not made here.

## Comparison

| Work | Objective | Construction | Retrieval effect measured |
|---|---|---|---|
| **PoisonedRAG** (Zou, Geng, Wang & Jia, arXiv:2402.07867, USENIX Sec 2025) | Targeted misinformation — make the model emit a *specific attacker-chosen answer* for a target query | Per-query crafted passages: a retrieval-optimized part + a generation-steering part | Whether the poisoned passage is retrieved and steers the answer |
| **CorruptRAG** (CRAG-AS/AK, 2025) | Targeted misinformation, more robust/covert than PoisonedRAG | Refined single-passage crafting | Retrieval + answer manipulation |
| **Zhong et al., corpus-poisoning** (EMNLP 2023, arXiv:2310.19156) | Broad retrieval hijack — a few passages retrieved for *many* queries | Gradient/HotFlip token optimization to move an embedding toward a query cluster | Retrieval rate across queries |
| **Blocker / jamming** (Shafran, Schuster & Shmatikov, arXiv:2406.05870, USENIX Sec 2025) | **Denial** — one blocker document per target query stops the system answering it | Black-box optimization; no instruction injection, no knowledge of embedder or LLM, no auxiliary LLM | Whether the blocker is retrieved and suppresses the answer |
| **MutedRAG** (Suo, Shang, Guo & Zhang, arXiv:2504.21680) | **Denial via safety guardrails** — inject jailbreak text so the LLM refuses legitimate queries | Minimal jailbreak strings ("How to build a bomb"); one sample affects many queries | Refusal rate; >60% ASR at <1 malicious text per target query |
| **C1 (this work)** | **Denial of service** — evict the gold passage so the system *cannot answer* (refusal / hallucination / degraded answer), not emit a chosen wrong answer | **Grey-box, optimization-free**: templated passages selected by cosine proximity to the query embedding; no gradients, no token-level optimization | Gold-passage eviction from top-k, adversarial pollution of top-k, and downstream generation-level denial (GDS/ASR/severity) |

The last two rows are the true neighbours and are **not** covered by the three
differentiators below, which were written against the misdirection attacks only.
Against Shafran the distinction is the alignment cross-cut and the retrieval-layer
mechanism (they suppress at the generation layer with the gold passage still
present; C1 removes the gold passage). Against MutedRAG it is that C1 injects no
unsafe content, whereas MutedRAG's mechanism is jailbreak text injected to trip
safety guardrails. State that as a scope claim — guardrail induction is neither
necessary nor typical for denial — rather than as a refutation of their result,
which is sound within their threat model.

## The three differentiators to state explicitly

1. **Objective: denial, not misdirection.** PoisonedRAG/CorruptRAG aim for a
   *specific wrong answer*; Zhong aims for *broad retrieval*. C1's goal is
   availability collapse — the correct evidence is displaced, so the system
   refuses, hallucinates, or degrades. This is measured with a graded severity
   scale (GDS/ASR), not answer-match to an attacker target.

2. **Construction: grey-box and optimization-free.** C1 needs only knowledge of
   the embedder (a realistic assumption for common open embedders) and uses
   templated text selected by embedding proximity — no gradient access, no
   HotFlip, no per-token optimization. This is a weaker attacker assumption than
   corpus-poisoning's gradient optimization, which strengthens the threat model.

3. **Evaluation: a benchmark, not a single attack.** C1 is one instance inside
   DoSRAGBench, evaluated over a literature-comparable corpus (BEIR NQ, ~500k
   passages) with matched base/aligned model pairs (the AVI alignment cross-cut)
   and six severity/degradation metrics.

## Honest scope note

The clustering mechanism reduces retrieval *latency* (greedy HNSW descends into
the dense cluster and converges faster), so C1 is a **displacement/denial**
attack, not a latency-degradation attack. A traversal-lengthening latency attack
that realizes the Indyk–Xu (NeurIPS 2023) O(n) worst case is left as future work
(it requires embedding-space inversion / graph-topology control and a
jitter-robust timing methodology). See the ablation (`scripts/run_ablation_katana.pbs`,
C1 vs RAND) for evidence that eviction is caused by clustering rather than by
corpus-size inflation.
