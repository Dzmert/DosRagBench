#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit every (model-pair × attack) generation run on Katana as parallel jobs,
# then compare them once they finish. One command, whole matrix.
#
#   bash scripts/submit_all.sh
#   NUM_QUERIES=1000 bash scripts/submit_all.sh
#   PAIRS="llama-3.1-8b qwen-2.5-7b" CATS="A1 C1" bash scripts/submit_all.sh
#   SKIP_PREP=1 bash scripts/submit_all.sh          # reuse existing data/
#
# Job DAG (so the shared 500k FAISS index is built once, not N times):
#
#   prep ──► warm-up run (builds index) ──►┬─ run 2 ─┐
#                                          ├─ run 3 ─┤─► compare (recompute + AVI)
#                                          └─ run N ─┘
#
# The warm-up is just the first run in the matrix; the rest depend on it so they
# reuse its cached index instead of racing to rebuild it. `compare` waits for ALL
# runs (afterany = even if some fail, you still get a table of whatever finished).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of where this is called from

# ── Matrix (override via env) ──
PAIRS="${PAIRS:-llama-3.1-8b qwen-2.5-7b mistral-7b llama-r1-8b}"
CATS="${CATS:-A1 C1}"
NUM_QUERIES="${NUM_QUERIES:-1000}"
KB_SIZE="${KB_SIZE:-500000}"
CORPUS="${CORPUS:-beir}"
SKIP_PREP="${SKIP_PREP:-0}"

echo "Pairs:       $PAIRS"
echo "Categories:  $CATS"
echo "Num queries: $NUM_QUERIES   KB size: $KB_SIZE"
echo

# ── 1. Data prep (one CPU job). Regenerates queries.json at the requested count
#       so downstream generation jobs all read the same, consistent dataset. ──
DEP_PREP=""
if [ "$SKIP_PREP" = "0" ]; then
    PREP=$(qsub - <<EOF
#PBS -N dosrag_prep
#PBS -l select=1:ncpus=8:mem=90gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o dosrag_prep.log
#PBS -M z5419867@ad.unsw.edu.au
#PBS -m ae
set -euo pipefail
cd "\$PBS_O_WORKDIR"
module load python/3.11 || true
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a
export HF_HOME="/srv/scratch/\$USER/hf_cache"
export HF_DATASETS_CACHE="/srv/scratch/\$USER/hf_cache/datasets"
export DOSRAGBENCH_CACHE="/srv/scratch/\$USER/dosragbench_cache"
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$DOSRAGBENCH_CACHE"
python scripts/prepare_data.py --corpus "$CORPUS" --num-queries "$NUM_QUERIES" --kb-size "$KB_SIZE"
rm -f data/.prepared_*   # stale run-guard markers no longer match the new query count
EOF
)
    echo "prep         -> $PREP"
    DEP_PREP="-W depend=afterok:$PREP"
fi

# ── 2. Generation runs. First one is the index-building warm-up; the rest wait
#       on it (afterany) so they reuse the cached index. ──
RUN_IDS=()
WARMUP=""
for PAIR in $PAIRS; do
    for CAT in $CATS; do
        if [ -z "$WARMUP" ]; then
            DEP="$DEP_PREP"
        else
            DEP="-W depend=afterany:$WARMUP"
        fi
        JID=$(qsub $DEP \
            -v PAIR="$PAIR",CATEGORY="$CAT",NUM_QUERIES="$NUM_QUERIES" \
            scripts/submit_katana.sh)
        RUN_IDS+=("$JID")
        [ -z "$WARMUP" ] && WARMUP="$JID"
        printf "run %-14s %-4s -> %s\n" "$PAIR" "$CAT" "$JID"
    done
done

# ── 3. Compare (CPU). Recompute metrics from raw outputs, then build the AVI
#       report across every run that finished. ──
DEP_ALL=$(IFS=:; echo "${RUN_IDS[*]}")
CMP=$(qsub -W depend=afterany:"$DEP_ALL" - <<'EOF'
#PBS -N dosrag_compare
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o dosrag_compare.log
#PBS -M z5419867@ad.unsw.edu.au
#PBS -m ae
set -euo pipefail
cd "$PBS_O_WORKDIR"
module load python/3.11 || true
source .venv/bin/activate
python scripts/recompute_metrics.py
python scripts/compute_avi.py
echo "Comparison written to results/avi_report.md"
EOF
)
echo "compare      -> $CMP"

echo
echo "Submitted $(( ${#RUN_IDS[@]} )) runs + compare. Watch: qstat -u \$USER"
echo "When done:  cat results/avi_report.md   (and results/avi_report.json)"
