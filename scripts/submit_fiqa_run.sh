#!/bin/bash
#PBS -N fiqa_run
#PBS -l select=1:ncpus=8:mem=90gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M z5419867@ad.unsw.edu.au
#PBS -m ae

# One (PAIR, CATEGORY) attack cell of the FiQA generalisation arm. Reads
# data_fiqa/ and writes results_fiqa/, so both the NQ headline grid (results/)
# and the demo's NQ data (data/) are left untouched.
#
# Submit from the repo root:
#   qsub -v PAIR=llama-3.1-8b,CATEGORY=D2 scripts/submit_fiqa_run.sh
#   qsub -v PAIR=qwen-2.5-7b,CATEGORY=C1,NUM_QUERIES=1000 scripts/submit_fiqa_run.sh

set -euo pipefail
cd "$PBS_O_WORKDIR"

module load python/3.11 cuda/12.1
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

export HF_HOME="/srv/scratch/$USER/hf_cache"
export DOSRAGBENCH_CACHE="/srv/scratch/$USER/dosragbench_cache"
mkdir -p "$HF_HOME" "$DOSRAGBENCH_CACHE"

PAIR=${PAIR:?"set PAIR (e.g. llama-3.1-8b)"}
CATEGORY=${CATEGORY:?"set CATEGORY (e.g. D2)"}
NUM_QUERIES=${NUM_QUERIES:-1000}

if [ ! -f data_fiqa/queries.json ]; then
    echo "data_fiqa/ not found -- run scripts/submit_fiqa_prep.sh first." >&2
    exit 1
fi

echo "FiQA run: pair=$PAIR category=$CATEGORY queries=$NUM_QUERIES"
python3 scripts/run_attack.py \
    --category "$CATEGORY" --model-pair "$PAIR" \
    --num-queries "$NUM_QUERIES" \
    --data-dir data_fiqa \
    --results-root results_fiqa \
    --device cuda

echo "Done. Results in results_fiqa/${PAIR}_${CATEGORY}/"
