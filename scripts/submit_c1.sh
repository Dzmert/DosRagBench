#!/bin/bash
#PBS -N DosRagC1
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -M z5419867@ad.unsw.edu.au
cd $PBS_O_WORKDIR
module load python/3.11 cuda/12.1
source .venv/bin/activate

set -a
source .env
set +a

export HF_HOME=/srv/scratch/$USER/hf_cache
export DOSRAGBENCH_CACHE=/srv/scratch/$USER/dosragbench_cache
export HF_HUB_OFFLINE=1          # everything is cached; don't phone home
mkdir -p "$HF_HOME" "$DOSRAGBENCH_CACHE"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

POLLUTION=${POLLUTION:-0.05}

python scripts/run_attack.py \
    --category C1 \
    --model-pair llama-3.1-8b \
    --c1-latency \
    --pollution-rate "$POLLUTION"
