#!/bin/bash
#PBS -N demo_rehearsal
#PBS -l select=1:ncpus=8:mem=90gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -M z5419867@ad.unsw.edu.au
#PBS -m ae

# Unattended rehearsal of the live demo (scripts/demo.py) on the NQ corpus.
#
# Two jobs:
#   1. WARM THE CACHE. demo.py caches its FAISS index to repo-local
#      .cache/index_<sig>.faiss. Building it is the slow part (minutes); once this
#      job runs, demo day loads it in seconds. The cache persists in the repo dir.
#   2. PROVE THE PIPELINE. Feeds canned questions on stdin so the whole
#      base-answers / aligned-refuses-under-attack path runs end to end on GPU,
#      no human at the keyboard. demo.py exits cleanly on EOF.
#
# This does NOT replace the LIVE tmux rehearsal (practising at the interactive
# prompt, testing disconnect survival) -- it de-risks the compute and the cache.
#
# Submit from the repo root:
#   qsub scripts/submit_demo_rehearsal.sh
# Override the pair/attack:
#   qsub -v PAIR=qwen-2.5-7b,ATTACK=D3 scripts/submit_demo_rehearsal.sh

set -euo pipefail
cd "$PBS_O_WORKDIR"

module load python/3.11 cuda/12.1
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

export HF_HOME="/srv/scratch/$USER/hf_cache"
export DOSRAGBENCH_CACHE="/srv/scratch/$USER/dosragbench_cache"
mkdir -p "$HF_HOME" "$DOSRAGBENCH_CACHE"

PAIR=${PAIR:-llama-3.1-8b}
ATTACK=${ATTACK:-D2}

if [ ! -f data/queries.json ]; then
    echo "data/ (NQ) not found -- this rehearsal runs against the NQ demo corpus." >&2
    exit 1
fi

echo "=================================================================="
echo "DEMO REHEARSAL  pair=$PAIR  attack=$ATTACK  corpus=data/ (NQ)"
echo "start: index build/load timing is the number that matters below."
echo "=================================================================="

# Canned session, one line per prompt, EOF ends it:
#   :suggest         -> prints 10 vetted known-good-retrieval questions (copy
#                       these for the live demo) into the log
#   <a free question> -> exercises the full clean+attacked x base+aligned path
#   :quit            -> clean exit
SECONDS=0
printf '%s\n' \
    ':suggest' \
    'who wrote the origin of species' \
    'when was the eiffel tower completed' \
    ':quit' \
  | python3 scripts/demo.py --model-pair "$PAIR" --attack "$ATTACK" --device cuda

echo "=================================================================="
echo "REHEARSAL DONE in ${SECONDS}s."
SIG_CACHE=$(ls -1 .cache/index_*.faiss 2>/dev/null | head -1 || true)
if [ -n "$SIG_CACHE" ]; then
    echo "CACHE OK: warm index left at $SIG_CACHE"
    echo "          -> demo day will LOAD this, not rebuild. Fast start confirmed."
else
    echo "CACHE WARNING: no .cache/index_*.faiss found -- demo day would rebuild."
fi
echo "=================================================================="
