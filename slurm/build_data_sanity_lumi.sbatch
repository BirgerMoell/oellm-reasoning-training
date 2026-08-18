#!/bin/bash
#SBATCH --job-name=oellm-reason-sanity-data
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
module purge

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${DATA_CONFIG:=configs/data/reasoning-sanity.yaml}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

mkdir -p logs "$OELLM_RUN_ROOT/cache/huggingface"
echo "Building sanity artifact from $DATA_CONFIG into $OELLM_RUN_ROOT"

singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  export HF_HOME=$OELLM_RUN_ROOT/cache/huggingface
  export HF_DATASETS_CACHE=$OELLM_RUN_ROOT/cache/huggingface/datasets
  export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd $SLURM_SUBMIT_DIR
  python3 scripts/build_mix.py --config $DATA_CONFIG --root $OELLM_RUN_ROOT --num-proc $SLURM_CPUS_PER_TASK
  python3 scripts/validate_run.py --root $OELLM_RUN_ROOT --config $DATA_CONFIG --full
"

echo "Sanity data build completed at $(date --iso-8601=seconds)"
