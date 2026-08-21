#!/bin/bash
#SBATCH --job-name=oellm-release-export
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/export-%j.out
#SBATCH --error=logs/export-%j.err

set -euo pipefail

: "${SELECTED_CHECKPOINT:?Set SELECTED_CHECKPOINT to the accepted checkpoint directory}"
: "${SOURCE_STEP:?Set SOURCE_STEP to the selected optimizer step}"
: "${RELEASE_DIR:?Set RELEASE_DIR to a new output directory}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  cd $SLURM_SUBMIT_DIR
  python3 scripts/export_release.py \
    --checkpoint '$SELECTED_CHECKPOINT' \
    --source-step '$SOURCE_STEP' \
    --output '$RELEASE_DIR'
"
