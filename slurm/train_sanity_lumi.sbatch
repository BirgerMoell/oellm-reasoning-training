#!/bin/bash
#SBATCH --job-name=oellm-reason-sanity
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

export TRAIN_CONFIG=configs/train/sanity.yaml
export GPUS_PER_NODE=8

# Execute the exact production launcher inside this shorter allocation.
exec bash slurm/train_lumi.sbatch
