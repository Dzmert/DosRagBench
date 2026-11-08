#!/bin/bash
# Submit the FiQA attack matrix as GPU jobs. The first job is the index warm-up;
# the rest depend on it (afterany) so the FiQA HNSW index (~57k docs) is built
# ONCE and reused -- the cache key is a hash of the doc-ids, so the FiQA corpus
# builds its own index and never collides with the cached NQ one.
#
# Everything writes to results_fiqa/ and reads data_fiqa/, so results/ and data/
# (the NQ demo) stay untouched.
#
# Default PAIRS includes llama-r1-8b for 4-family parity with the NQ grid. Drop
# it (PAIRS="llama-3.1-8b qwen-2.5-7b mistral-7b") if the R1 wall-clock is too
# costly -- the other three families are unaffected.
#
#   bash scripts/submit_fiqa_all.sh
#   PAIRS="llama-3.1-8b" CATS="D2 D3" bash scripts/submit_fiqa_all.sh
#   NUM_QUERIES=1000 bash scripts/submit_fiqa_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PAIRS="${PAIRS:-llama-3.1-8b qwen-2.5-7b mistral-7b llama-r1-8b}"
CATS="${CATS:-A1 A2 A3 B1 B2 B3 C1 C2 C3 D1 D2 D3 D4}"
NUM_QUERIES="${NUM_QUERIES:-1000}"

if [ ! -f data_fiqa/queries.json ]; then
    echo "data_fiqa/ not found -- run scripts/submit_fiqa_prep.sh first." >&2
    exit 1
fi

echo "Pairs:      $PAIRS"
echo "Categories: $CATS"
echo "Queries:    $NUM_QUERIES"
echo

RUN_IDS=()
WARMUP=""
for PAIR in $PAIRS; do
  for CAT in $CATS; do
    if [ -z "$WARMUP" ]; then
        DEP=""
    else
        DEP="-W depend=afterany:$WARMUP"
    fi
    JID=$(qsub $DEP -v PAIR="$PAIR",CATEGORY="$CAT",NUM_QUERIES="$NUM_QUERIES" \
        scripts/submit_fiqa_run.sh)
    RUN_IDS+=("$JID")
    [ -z "$WARMUP" ] && WARMUP="$JID"
    printf "run %-14s %-4s -> %s\n" "$PAIR" "$CAT" "$JID"
  done
done

echo
echo "Submitted ${#RUN_IDS[@]} FiQA runs (warm-up: $WARMUP)."
echo "Watch:   qstat -u \$USER"
echo "Results: results_fiqa/<pair>_<attack>/raw_results.json"
