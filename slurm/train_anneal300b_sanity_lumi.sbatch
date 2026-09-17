#!/bin/bash
#SBATCH --job-name=oellm-a300b-rsn-sanity
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
DATA=$OELLM_RUN_ROOT/data/reasoning-anneal300b-v1/train.parquet
MANIFEST=$OELLM_RUN_ROOT/data/reasoning-anneal300b-v1/manifest.json
OUTPUT=$OELLM_RUN_ROOT/checkpoints/reasoning-anneal300b-sanity
for required in "$DATA" "$MANIFEST"; do
  if [[ ! -s "$required" ]]; then
    echo "Sanity preflight failed: missing or empty $required" >&2
    exit 2
  fi
done
if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Sanity preflight failed: output is not empty: $OUTPUT" >&2
  exit 2
fi

export TRAIN_CONFIG=configs/train/reasoning-anneal300b-sanity.yaml
export CHAT_TEMPLATE_FILE=templates/oellm_qwen3_assistant_mask.jinja
export GPUS_PER_NODE=8

# The only differences from production are max_steps, checkpoint interval, and output path.
exec bash slurm/train_lumi.sbatch
