#!/bin/bash
#SBATCH --job-name=oellm-repeat-audit
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${MAX_PER_SOURCE_LANGUAGE:=2000}"

CONTAINER=${OELLM_CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
OUTPUT=$OELLM_RUN_ROOT/audits/repetition/repetition-audit.json

mkdir -p logs "$(dirname "$OUTPUT")"
singularity exec -B "$BIND" "$CONTAINER" python3 scripts/audit_repetition.py \
  --examples model-card/examples.json \
  --parquet "$OELLM_RUN_ROOT/data/reasoning-v1/train.parquet" \
  --manifest "$OELLM_RUN_ROOT/data/reasoning-v1/manifest.json" \
  --max-per-source-language "$MAX_PER_SOURCE_LANGUAGE" \
  --output "$OUTPUT"

echo "Repetition audit completed at $(date --iso-8601=seconds): $OUTPUT"
