#!/bin/bash
# LUMI login-node wrapper for staging pinned Hugging Face snapshots.

set -euo pipefail

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${DATA_CONFIG:=configs/data/reasoning-v1.yaml}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray

mkdir -p logs "$OELLM_RUN_ROOT/cache/huggingface"
scripts/install_lumi_dependencies.sh
singularity exec -B "$BIND" "$CONTAINER" env \
  PYTHONPATH="$OVERLAY" \
  HF_HOME="$OELLM_RUN_ROOT/cache/huggingface" \
  HF_DATASETS_CACHE="$OELLM_RUN_ROOT/cache/huggingface/datasets" \
  python3 scripts/stage_hf.py \
    --config "$DATA_CONFIG" \
    --root "$OELLM_RUN_ROOT" \
    "$@"
