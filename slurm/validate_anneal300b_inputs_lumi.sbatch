#!/bin/bash
#SBATCH --job-name=oellm-a300b-inputs
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
CONTAINER=${OELLM_CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

mkdir -p logs
singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  export HF_HOME=$OELLM_RUN_ROOT/cache/huggingface
  export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd $SLURM_SUBMIT_DIR
  python3 scripts/validate_chat_template.py \
    --config configs/data/reasoning-anneal300b-v1.yaml \
    --root $OELLM_RUN_ROOT
"
