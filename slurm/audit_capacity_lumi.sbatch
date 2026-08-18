#!/bin/bash
#SBATCH --job-name=oellm-capacity
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=480G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
module purge

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${DATA_CONFIG:=configs/data/reasoning-v1.yaml}"
: "${SOURCE_IDS:?Set SOURCE_IDS to comma-separated configured source IDs}"
: "${SEED_DEDUP:=}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

mkdir -p logs
IFS=',' read -r -a SOURCE_ARRAY <<< "$SOURCE_IDS"
SOURCE_ARGS=()
for source_id in "${SOURCE_ARRAY[@]}"; do
  SOURCE_ARGS+=(--source "$source_id")
done
if [[ -n "$SEED_DEDUP" ]]; then
  SOURCE_ARGS+=(--seed-dedup "$SEED_DEDUP")
fi

singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  export HF_HOME=$OELLM_RUN_ROOT/cache/huggingface
  export HF_DATASETS_CACHE=$OELLM_RUN_ROOT/cache/huggingface/datasets
  export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd $SLURM_SUBMIT_DIR
  python3 scripts/audit_source_capacity.py --config $DATA_CONFIG --root $OELLM_RUN_ROOT \
    --num-proc $SLURM_CPUS_PER_TASK ${SOURCE_ARGS[*]}
"
