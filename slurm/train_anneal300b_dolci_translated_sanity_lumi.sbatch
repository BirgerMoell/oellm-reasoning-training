#!/bin/bash
#SBATCH --job-name=oellm-a300b-dolci-sanity
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
DATA=$OELLM_RUN_ROOT/data/reasoning-anneal300b-dolci-translated-v2/train.parquet
MANIFEST=$OELLM_RUN_ROOT/data/reasoning-anneal300b-dolci-translated-v2/manifest.json
OUTPUT=$OELLM_RUN_ROOT/checkpoints/reasoning-anneal300b-dolci-translated-sanity
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

export TRAIN_CONFIG=configs/train/reasoning-anneal300b-dolci-translated-sanity.yaml
export CHAT_TEMPLATE_FILE=templates/oellm_qwen3_assistant_mask.jinja
export GPUS_PER_NODE=8

exec bash slurm/train_lumi.sbatch
