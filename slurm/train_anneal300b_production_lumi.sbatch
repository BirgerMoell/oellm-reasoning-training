#!/bin/bash
#SBATCH --job-name=oellm-a300b-reason-v1
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-05:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${RESUME_FROM_CHECKPOINT:=0}"
DATA=$OELLM_RUN_ROOT/data/reasoning-anneal300b-v1/train.parquet
MANIFEST=$OELLM_RUN_ROOT/data/reasoning-anneal300b-v1/manifest.json
OUTPUT=$OELLM_RUN_ROOT/checkpoints/reasoning-anneal300b-v1
for required in "$DATA" "$MANIFEST"; do
  if [[ ! -s "$required" ]]; then
    echo "Production preflight failed: missing or empty $required" >&2
    exit 2
  fi
done
if [[ "$RESUME_FROM_CHECKPOINT" == "0" ]] && \
   [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Production preflight failed: output is not empty: $OUTPUT" >&2
  exit 2
fi
if [[ "$RESUME_FROM_CHECKPOINT" != "0" ]] && [[ ! -d "$OUTPUT" ]]; then
  echo "Production preflight failed: resume requested but output does not exist: $OUTPUT" >&2
  exit 2
fi

export TRAIN_CONFIG=configs/train/reasoning-anneal300b-v1.yaml
export CHAT_TEMPLATE_FILE=templates/oellm_qwen3_assistant_mask.jinja
export GPUS_PER_NODE=8
export RESUME_FROM_CHECKPOINT

exec bash slurm/train_lumi.sbatch
