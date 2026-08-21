#!/bin/bash
#SBATCH --job-name=oellm-release-check
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/release-check-%j.out
#SBATCH --error=logs/release-check-%j.err

set -euo pipefail

: "${RELEASE_DIR:?Set RELEASE_DIR to the exported model directory}"
: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --rocm -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  cd $SLURM_SUBMIT_DIR
  python3 scripts/validate_release.py '$RELEASE_DIR' --gpu-smoke
  python3 scripts/check_checkpoint_finite.py '$RELEASE_DIR' --full-scan --chunk-elements 4194304
"
