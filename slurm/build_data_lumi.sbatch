#!/bin/bash
#SBATCH --job-name=oellm-reason-data
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

# The build executes entirely inside the pinned container. Host `module purge`
# can block in Lmod/Lustre metadata I/O on LUMI CPU nodes before useful work starts.

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${DATA_CONFIG:=configs/data/reasoning-v1.yaml}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

mkdir -p logs "$OELLM_RUN_ROOT/cache/huggingface"
echo "Building $DATA_CONFIG into $OELLM_RUN_ROOT"

RUN_DIR=$OELLM_RUN_ROOT/runs/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
cp "$DATA_CONFIG" "$RUN_DIR/resolved-data-config.yaml"
git rev-parse HEAD > "$RUN_DIR/repository-git-sha.txt"
git status --porcelain=v1 > "$RUN_DIR/repository-status.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$RUN_DIR/slurm-job.txt"
sha256sum "$RUN_DIR/resolved-data-config.yaml" > "$RUN_DIR/provenance-sha256.txt"
export REPOSITORY_GIT_SHA
REPOSITORY_GIT_SHA=$(git rev-parse HEAD)

singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  export HF_HOME=$OELLM_RUN_ROOT/cache/huggingface
  export HF_DATASETS_CACHE=$OELLM_RUN_ROOT/cache/huggingface/datasets
  export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export REPOSITORY_GIT_SHA=$REPOSITORY_GIT_SHA
  cd $SLURM_SUBMIT_DIR
  python3 scripts/build_mix.py --config $DATA_CONFIG --root $OELLM_RUN_ROOT --num-proc $SLURM_CPUS_PER_TASK
  python3 scripts/validate_run.py --root $OELLM_RUN_ROOT --config $DATA_CONFIG --full
"

echo "Data build completed at $(date --iso-8601=seconds)"
