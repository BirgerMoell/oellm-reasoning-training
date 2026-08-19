#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 DATA_JOB_ID SOURCE_ARTIFACT_ROOT EXPECTED_BUILD_SHA" >&2
  exit 2
fi

DATA_JOB_ID=$1
SOURCE_ARTIFACT_ROOT=$2
EXPECTED_BUILD_SHA=$3
OELLM_RUN_ROOT=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts
SMOKE_OUTPUT=$OELLM_RUN_ROOT/checkpoints/reasoning-v1-smoke
PRODUCTION_OUTPUT=$OELLM_RUN_ROOT/checkpoints/reasoning-v1

if ! [[ "$DATA_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "DATA_JOB_ID must be numeric" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ARTIFACT_ROOT" ]]; then
  echo "Missing isolated artifact root: $SOURCE_ARTIFACT_ROOT" >&2
  exit 2
fi
if [[ -e "$OELLM_RUN_ROOT/data/reasoning-v1" ]]; then
  echo "Canonical reasoning-v1 data destination already exists" >&2
  exit 2
fi
for output in "$SMOKE_OUTPUT" "$PRODUCTION_OUTPUT"; do
  if [[ -e "$output" ]]; then
    echo "Refusing existing checkpoint output: $output" >&2
    exit 2
  fi
done

data_state=$(sacct -X -j "$DATA_JOB_ID" --format=State -n -P | head -n 1 | cut -d'|' -f1)
data_state=${data_state%%+}
case "$data_state" in
  COMPLETED)
    data_dependency=()
    ;;
  PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED)
    data_dependency=(--dependency="afterok:$DATA_JOB_ID")
    ;;
  *)
    echo "Data job $DATA_JOB_ID is not promotable: state=$data_state" >&2
    exit 2
    ;;
esac

promotion_job=$(sbatch --parsable --kill-on-invalid-dep=yes \
  "${data_dependency[@]}" \
  --export="ALL,SOURCE_ARTIFACT_ROOT=$SOURCE_ARTIFACT_ROOT,EXPECTED_BUILD_SHA=$EXPECTED_BUILD_SHA,OELLM_RUN_ROOT=$OELLM_RUN_ROOT" \
  slurm/promote_data_lumi.sbatch)

smoke_job=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$promotion_job" \
  --nodes=1 --gpus-per-node=8 --time=0-02:00:00 \
  --export="ALL,GPUS_PER_NODE=8,OELLM_RUN_ROOT=$OELLM_RUN_ROOT,TRAIN_CONFIG=configs/train/smoke.yaml" \
  slurm/train_lumi.sbatch)

audit_job=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$smoke_job" \
  --export="ALL,CHECKPOINT=$SMOKE_OUTPUT/checkpoint-10,FULL_SCAN=1" \
  slurm/check_checkpoint_lumi.sbatch)

production_job=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$audit_job" \
  --export="ALL,OELLM_RUN_ROOT=$OELLM_RUN_ROOT" \
  slurm/train_production_lumi.sbatch)

mkdir -p "$OELLM_RUN_ROOT/runs"
record=$OELLM_RUN_ROOT/runs/pipeline-$DATA_JOB_ID.txt
{
  echo "data_job=$DATA_JOB_ID"
  echo "data_state=$data_state"
  echo "promotion_job=$promotion_job"
  echo "smoke_job=$smoke_job"
  echo "audit_job=$audit_job"
  echo "production_job=$production_job"
  echo "source_artifact_root=$SOURCE_ARTIFACT_ROOT"
  echo "expected_build_sha=$EXPECTED_BUILD_SHA"
} | tee "$record"
