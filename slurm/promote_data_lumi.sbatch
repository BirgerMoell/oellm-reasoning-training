#!/bin/bash
#SBATCH --job-name=oellm-data-promote
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

: "${SOURCE_ARTIFACT_ROOT:?Set SOURCE_ARTIFACT_ROOT to the isolated validated build root}"
: "${EXPECTED_BUILD_SHA:?Set EXPECTED_BUILD_SHA to the data manifest repository SHA}"
: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"

SOURCE=$SOURCE_ARTIFACT_ROOT/data/reasoning-v1
DESTINATION=$OELLM_RUN_ROOT/data/reasoning-v1
RUN_DIR=$OELLM_RUN_ROOT/runs/$SLURM_JOB_ID

mkdir -p logs "$RUN_DIR"
python3 scripts/promote_data.py \
  --source "$SOURCE" \
  --destination "$DESTINATION" \
  --expected-repository-sha "$EXPECTED_BUILD_SHA" \
  | tee "$RUN_DIR/promotion.txt"
git rev-parse HEAD > "$RUN_DIR/repository-git-sha.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$RUN_DIR/slurm-job.txt"
