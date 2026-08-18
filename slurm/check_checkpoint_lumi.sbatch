#!/bin/bash
#SBATCH --job-name=oellm-ckpt-check
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=220G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
module purge

: "${CHECKPOINT:?Set CHECKPOINT to a checkpoint directory or model safetensors file}"
: "${FULL_SCAN:=1}"
: "${EXPECTED_TENSORS:=399}"
: "${EXPECTED_VALUES:=9101947904}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

if [[ ! -e "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi

mkdir -p logs
ARGS=()
if [[ "$FULL_SCAN" == "1" ]]; then
  ARGS+=(--full)
fi

singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  cd $SLURM_SUBMIT_DIR
  python3 scripts/check_checkpoint_finite.py ${ARGS[*]} \
    --expected-tensors '$EXPECTED_TENSORS' --expected-values '$EXPECTED_VALUES' '$CHECKPOINT'
"
