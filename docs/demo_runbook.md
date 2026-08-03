# Thesis B demo — runbook

A live demo is mandatory for the Thesis B presentation. Bao's tiers: a video is
low effect, running code at the command line is mid, and **the assessor supplying
an input and the system running live is high**. This runbook targets the high tier.

`scripts/demo.py` does it. One assessor question shows the whole thesis:

| | no attack | under attack |
|---|---|---|
| **Base model** | answers | still answers |
| **Aligned model** | answers | *"the context does not support an answer"* |

Why this is feasible when a full run is GPU-hours: the hours come from 1,000
queries × 4 model pairs. **One** query is retrieval in milliseconds plus four
generations. Models and the FAISS index load once and stay resident — so the cost
is paid before the talk, not during it.

---

## The 20 minutes before you present

Start this while the room is filling. Load takes minutes; questions take seconds.

```bash
# 1. Interactive GPU session. Ask for longer than you need — you cannot extend it.
qsub -I -l select=1:ncpus=8:ngpus=1:mem=90gb -l walltime=03:00:00

# 2. Same environment the batch jobs use
cd $HOME/DosRagBench
module load python/3.11 cuda/12.1
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/srv/scratch/$USER/hf_cache}

# 3. Start the demo and LEAVE IT AT THE PROMPT
python3 scripts/demo.py --model-pair llama-3.1-8b --attack D2
```

Wait for `Ready in NNNs`. Ask one throwaway question yourself to confirm
generation works, then stop touching it.

**Use `tmux` or `screen` before step 3.** A dropped SSH connection otherwise ends
the demo and there is no time to reload models in front of an assessor.

```bash
tmux new -s demo     # then start the demo inside it
# if the connection drops:  ssh back in,  tmux attach -t demo
```

---

## Driving it

| you type | what happens |
|---|---|
| any question | full run: clean retrieval → both models → inject → re-retrieve → both models |
| `:attack D3` | switch attack between questions |
| `:attacks` | the 8 worth demoing, with one-line notes |
| `:suggest` | 10 dataset questions with known-good retrieval — the safety net |
| `:help` `:quit` | |

Ctrl-C abandons a question and returns to the prompt; it does not kill the session.

### Which attack to ask for

* **D2 Circular Reference Chains** — the default. Strongest signal in the
  benchmark (c:b ≈ 7), so it is the most likely to visibly flip the aligned model.
* **D3 Epistemic Uncertainty** — the mechanism attack. Best if you want the
  narration to match the thesis claim word for word.
* **C1 Embedding Clustering** — the retrieval-layer attack, and the best *visual*:
  the gold passage disappears from the top-5 entirely and all five slots turn red.
  It needs no LLM at all, which also makes it the no-GPU fallback.

---

## When a question goes badly — say this, do not apologise

**If the aligned model refuses before any attack**, retrieval was poor for that
question. This is not a broken demo, it is §3.7 of the thesis happening live:

> "Look at the retrieval panel — the gold passage isn't in the top five. My
> binning result says the alignment gap grows as retrieval degrades, so this model
> declines *before* I've attacked it at all. That's the mechanism, not the attack."

You have the numbers to back it: aligned clean-denial runs 0.212 at gold-rank 0 up
to 0.637 when gold isn't retrieved, while base models stay flat near 0.02.

**If neither model flips**, that is expected — roughly one attacked query in three
flips, and the demo says so on screen. Switch attack (`:attack D2`) or take another
question. Do not run the same question twice hoping for a different answer:
decoding is greedy and deterministic, so it will be identical, and an assessor who
notices that will wonder what else is non-reproducible.

**If a generation errors**, the panel prints the error and the prompt returns. Keep
going; one failed generation is not a failed demo.

---

## Fallbacks, in order

1. **`--no-llm`** — retrieval only, no GPU, no model load, starts in seconds:
   ```bash
   python3 scripts/demo.py --no-llm --attack C1
   ```
   Still assessor-driven and still live, and C1 evicting the gold passage from the
   top-5 is the single clearest visual in the project. Mid-to-high effect.
2. **A smaller pair** if VRAM is short — `--model-pair qwen-2.5-7b`. Say plainly
   that it is a different pair from the headline result.
3. **A recorded screen capture** of a successful run. Record one the day before
   *regardless* — it costs ten minutes and it is the difference between a hiccup
   and a failed assessment criterion. Low effect on its own; fine as insurance.

---

## Rehearse twice, on the real machine

Demos fail on cold caches, expired sessions and OOM — not on logic. Both rehearsals
must be on Katana, in a fresh interactive session, from a clean shell.

Checklist:

- [ ] Interactive session obtained without a long queue wait (try at the time of
      day you will present — queue depth varies)
- [ ] `data/knowledge_base.json` is real data, not an LFS pointer (the demo checks
      this and tells you, but find out now rather than then)
- [ ] The FAISS index loads from cache rather than rebuilding — the log line says
      which. A rebuild at 500k passages is far too slow to do live.
- [ ] Both models load and a throwaway question completes
- [ ] At least one question visibly flips the aligned model but not the base model
- [ ] `tmux` session survives a deliberate disconnect
- [ ] Terminal font size legible from the back of the room, and the window is wide
      — the retrieval table adapts to width, but wider is better

---

## What to say while it runs

Keep narrating; generation takes seconds and silence is longer than it sounds.

1. *"The corpus is half a million Wikipedia passages. Retrieval is HNSW, the same
   index structure Pinecone and Weaviate use."* — while retrieval prints.
2. *"Both models get identical context. The only difference is instruction tuning."*
   — while the clean answers generate.
3. *"Now I inject the adversarial passages. Watch the retrieval panel — they take
   four of the five slots."* — after injection.
4. *"Same question, same model, same context budget. The base model still answers.
   The aligned one won't."* — on the summary table.

Then land it: **the aligned model has not been jailbroken and has not produced
anything unsafe — it has been silenced, and that is an availability failure.**
