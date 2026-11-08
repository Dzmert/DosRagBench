#!/bin/bash
#PBS -N hotpot_run
#PBS -l select=1:ncpus=8:mem=90gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M z5419867@ad.unsw.edu.au
#PBS -m ae

# Step 3 of the HotpotQA repair: run ONE (PAIR, CATEGORY) attack against the
# rebuilt data_hotpotqa/ corpus. Writes to results_hotpotqa/ and reads
# data_hotpotqa/, so BOTH the NQ headline grid (results/) and the demo's NQ data
# (data/) are left untouched.
#
# Submit from the repo root:
#   qsub -v PAIR=llama-3.1-8b,CATEGORY=D2 scripts/submit_hotpot_run.sh
#   qsub -v PAIR=qwen-2.5-7b,CATEGORY=C1,NUM_QUERIES=1000 scripts/submit_hotpot_run.sh

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

if [ ! -f data_hotpotqa/queries.json ]; then
    echo "data_hotpotqa/ not found -- run scripts/submit_hotpot_prep.sh first." >&2
    exit 1
fi

echo "HotpotQA run: pair=$PAIR category=$CATEGORY queries=$NUM_QUERIES"
python3 scripts/run_attack.py \
    --category "$CATEGORY" --model-pair "$PAIR" \
    --num-queries "$NUM_QUERIES" \
    --data-dir data_hotpotqa \
    --results-root results_hotpotqa \
    --device cuda

echo "Done. Results in results_hotpotqa/${PAIR}_${CATEGORY}/"
