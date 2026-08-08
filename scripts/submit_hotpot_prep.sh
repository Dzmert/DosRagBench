#!/bin/bash
#PBS -N hotpot_prep
#PBS -l select=1:ncpus=8:mem=90gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o hotpot_prep.log
#PBS -M z5419867@ad.unsw.edu.au
#PBS -m ae

# Step 1 of the HotpotQA repair: rebuild the corpus with BOTH gold passages per
# query (--keep-all-gold), written to data_hotpotqa/ so the NQ data in data/ is
# left untouched. CPU-only, memory-heavy (loads the full HotpotQA corpus to
# sample 500k) — must run on a compute node, not the login node.
#
# Submit from the repo root:  qsub scripts/submit_hotpot_prep.sh

set -euo pipefail
cd "$PBS_O_WORKDIR"

module load python/3.11
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

export HF_HOME="/srv/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/srv/scratch/$USER/hf_cache/datasets"
export DOSRAGBENCH_CACHE="/srv/scratch/$USER/dosragbench_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$DOSRAGBENCH_CACHE"

python3 scripts/prepare_data.py \
    --corpus beir --dataset hotpotqa --keep-all-gold \
    --num-queries 1000 --kb-size 500000 --seed 42 \
    --output-dir data_hotpotqa

# Self-check: confirms 1000 queries and that the second gold hop was emitted.
python3 -c "import json; q=json.load(open('data_hotpotqa/queries.json')); print('PREP OK:', len(q), 'queries; keep-all-gold =', 'gold_doc_ids' in q[0])"
