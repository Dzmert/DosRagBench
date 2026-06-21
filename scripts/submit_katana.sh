#!/bin/bash
#PBS -N DosRagBench
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_id=A100
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M z5419867@student.unsw.edu.au

# DoSRAGBench — Katana HPC submission template.
#
# Usage:
#   qsub -v PAIR=llama-3.1-8b,CATEGORY=A1 scripts/submit_katana.sh
#   qsub -v PAIR=llama-r1-8b,CATEGORY=C1,NUM_QUERIES=20 scripts/submit_katana.sh
#
# For 70B models, bump to 2 GPUs:
#   qsub -l select=1:ncpus=16:mem=128gb:ngpus=2 -v PAIR=llama-70b,CATEGORY=A1 scripts/submit_katana.sh

set -euo pipefail

cd $PBS_O_WORKDIR

# ── Environment setup ──
module load python/3.11 cuda/12.1
source .venv/bin/activate

# Hugging Face cache on scratch to avoid quota issues
export HF_HOME=${HF_HOME:-/srv/scratch/$USER/hf_cache}
export DOSRAGBENCH_CACHE=${DOSRAGBENCH_CACHE:-/srv/scratch/$USER/dosragbench_cache}
mkdir -p "$HF_HOME" "$DOSRAGBENCH_CACHE"

# ── Parameters ──
PAIR=${PAIR:?"Must set PAIR (e.g. llama-3.1-8b)"}
CATEGORY=${CATEGORY:?"Must set CATEGORY (e.g. A1)"}
NUM_QUERIES=${NUM_QUERIES:-50}

echo "=========================================="
echo "DoSRAGBench run"
echo "  Pair:     $PAIR"
echo "  Category: $CATEGORY"
echo "  Queries:  $NUM_QUERIES"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# ── Ensure data exists ──
if [ ! -f data/queries.json ]; then
    echo "Preparing dataset..."
    python scripts/prepare_data.py --num-queries $NUM_QUERIES --kb-size 1000
fi

# ── Run attack ──
python scripts/run_attack.py \
    --category "$CATEGORY" \
    --model-pair "$PAIR" \
    --num-queries "$NUM_QUERIES"

echo "Done. Results in results/${PAIR}_${CATEGORY}/"
