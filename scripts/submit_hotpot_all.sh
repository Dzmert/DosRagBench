#!/bin/bash
# Submit the HotpotQA attack matrix as GPU jobs. The first job is the index
# warm-up; the rest depend on it (afterany) so the 500k HNSW index is built ONCE
# and reused -- the cache key is a hash of the doc-ids, so the HotpotQA corpus
# builds its own index and never collides with the cached NQ one.
#
# Everything writes to results_hotpotqa/ and reads data_hotpotqa/, so results/
# and data/ (the NQ demo) stay untouched.
#
#   bash scripts/submit_hotpot_all.sh
#   PAIRS="llama-3.1-8b" CATS="D2 D3" bash scripts/submit_hotpot_all.sh
#   NUM_QUERIES=1000 bash scripts/submit_hotpot_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PAIRS="${PAIRS:-llama-3.1-8b qwen-2.5-7b mistral-7b}"
CATS="${CATS:-A1 A2 A3 B1 B2 B3 C1 C2 C3 D1 D2 D3 D4}"
NUM_QUERIES="${NUM_QUERIES:-1000}"

if [ ! -f data_hotpotqa/queries.json ]; then
    echo "data_hotpotqa/ not found -- run scripts/submit_hotpot_prep.sh first." >&2
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
        scripts/submit_hotpot_run.sh)
    RUN_IDS+=("$JID")
    [ -z "$WARMUP" ] && WARMUP="$JID"
    printf "run %-14s %-4s -> %s\n" "$PAIR" "$CAT" "$JID"
  done
done

echo
echo "Submitted ${#RUN_IDS[@]} HotpotQA runs (warm-up: $WARMUP)."
echo "Watch:   qstat -u \$USER"
echo "Results: results_hotpotqa/<pair>_<attack>/raw_results.json"
