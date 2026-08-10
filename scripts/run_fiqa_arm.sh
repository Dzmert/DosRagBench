#!/bin/bash
# Driver for the FiQA generalisation arm. Run ONE phase at a time from the repo
# root on the Katana LOGIN node -- the phases that need a GPU/compute node are
# submitted via qsub, the light ones run inline. The pilot gate is deliberate:
# do NOT launch the full matrix until the pilot's answerable rate looks right
# (~0.6-0.7, decisively not ~0.24 like the withdrawn HotpotQA arm).
#
#   bash scripts/run_fiqa_arm.sh prep      # 1. build data_fiqa/ (qsub, CPU)
#   bash scripts/run_fiqa_arm.sh check     # 2. single-hop sanity (inline)
#   bash scripts/run_fiqa_arm.sh pilot     # 3. one cell (qsub, GPU)  <-- GATE
#   bash scripts/run_fiqa_arm.sh matrix    # 4. full matrix (qsub, GPU)
#   bash scripts/run_fiqa_arm.sh analyse   # 5. significance + figures (inline)
#
# Override the pilot cell or matrix scope with env vars, e.g.
#   PAIRS="llama-3.1-8b qwen-2.5-7b mistral-7b" bash scripts/run_fiqa_arm.sh matrix
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PHASE="${1:-}"

case "$PHASE" in
  prep)
    echo ">> Submitting FiQA data prep (CPU)."
    qsub scripts/submit_fiqa_prep.sh
    echo "   When it finishes, check fiqa_prep.log for 'PREP OK: N queries; single-gold = True'"
    echo "   then run:  bash scripts/run_fiqa_arm.sh check"
    ;;

  check)
    echo ">> Single-hop sanity on FiQA qrels (inline, login-node safe)."
    python3 scripts/check_gold_multiplicity.py --dataset fiqa
    if [ -f data_fiqa/queries.json ]; then
      python3 -c "import json; q=json.load(open('data_fiqa/queries.json')); print('data_fiqa OK:', len(q), 'queries; single-gold =', 'gold_doc_ids' not in q[0])"
    else
      echo "   (data_fiqa/ not built yet -- run the prep phase first.)"
    fi
    echo "   Expect: mean gold/query ~1.0-1.1, ragged distribution, VERDICT: usable."
    echo "   Then run:  bash scripts/run_fiqa_arm.sh pilot"
    ;;

  pilot)
    PAIR="${PAIR:-llama-3.1-8b}"
    CATEGORY="${CATEGORY:-D2}"
    echo ">> Submitting PILOT cell ${PAIR} x ${CATEGORY} (GPU). This is the gate."
    qsub -v PAIR="$PAIR",CATEGORY="$CATEGORY",NUM_QUERIES="${NUM_QUERIES:-1000}" scripts/submit_fiqa_run.sh
    echo "   When it finishes, inspect it:"
    echo "     python3 scripts/check_pilot.py results_fiqa/${PAIR}_${CATEGORY}"
    echo "   GATE: clean-baseline answerable must be ~0.6-0.7 (NOT ~0.24). Only then:"
    echo "     bash scripts/run_fiqa_arm.sh matrix"
    ;;

  matrix)
    echo ">> Submitting the full FiQA attack matrix (GPU)."
    echo "   (Default PAIRS includes llama-r1-8b; override PAIRS= to drop it.)"
    bash scripts/submit_fiqa_all.sh
    echo "   Watch:  qstat -u \$USER"
    echo "   When all cells finish:  bash scripts/run_fiqa_arm.sh analyse"
    ;;

  analyse)
    echo ">> Folding FiQA into significance (per-dataset FDR; NQ headline unchanged)."
    python3 scripts/compute_significance.py \
        --results-dir results --results-dir results_fiqa
    echo ">> Rebuilding the clean report (NQ + FiQA)."
    python3 scripts/clean_avi_report.py \
        --results-dir results --results-dir results_fiqa
    echo ">> Rendering FiQA figures (NQ figures untouched)."
    DATASET=FiQA python3 scripts/plot_forest.py
    DATASET=FiQA python3 scripts/plot_verdicts.py
    DATASET=FiQA python3 scripts/plot_asr_summary.py
    echo ">> Done. FiQA rows are in results/avi_significance.{md,json};"
    echo "   figures: results/{forest_genuine,verdict_breakdown,asr_risk_summary}_fiqa.png"
    ;;

  *)
    echo "Usage: bash scripts/run_fiqa_arm.sh {prep|check|pilot|matrix|analyse}" >&2
    echo "Run one phase at a time; see the header of this script for the flow." >&2
    exit 1
    ;;
esac
