#!/bin/bash
#PBS -N fiqa_prep
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o fiqa_prep.log
#PBS -M z5419867@ad.unsw.edu.au
#PBS -m ae

# Step 1 of the FiQA generalisation arm: build the corpus + queries for FiQA
# (BEIR financial-domain QA), written to data_fiqa/ so the NQ data in data/ is
# left untouched. FiQA is single-hop, so NO --keep-all-gold (unlike HotpotQA):
# each query keeps its single best gold passage, exactly the NQ path.
#
# FiQA's whole corpus is only ~57k passages, so --kb-size is set to 60000
# (500000 would be silently capped to the corpus and falsely imply a 500k
# haystack). Light on memory; a compute node is used only to keep off the login
# node while downloading.
#
# Submit from the repo root:  qsub scripts/submit_fiqa_prep.sh

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
    --corpus beir --dataset fiqa \
    --num-queries 1000 --kb-size 60000 --seed 42 \
    --output-dir data_fiqa

# Self-check: FiQA is single-hop, so confirm NO second-gold field was emitted
# (the opposite of the HotpotQA keep-all-gold check). Query count will be < 1000
# (FiQA has ~648 test queries) -- that is expected, not an error.
python3 -c "import json; q=json.load(open('data_fiqa/queries.json')); print('PREP OK:', len(q), 'queries; single-gold =', 'gold_doc_ids' not in q[0])"
