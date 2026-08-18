#!/bin/bash
#SBATCH --job-name=oellm-reason-v1
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-14:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

export TRAIN_CONFIG=configs/train/reasoning-v1.yaml
export GPUS_PER_NODE=8
: "${RESUME_FROM_CHECKPOINT:=0}"
export RESUME_FROM_CHECKPOINT

# Execute the tested launcher with a fixed production topology and recipe.
exec bash slurm/train_lumi.sbatch
