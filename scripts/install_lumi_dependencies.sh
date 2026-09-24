#!/bin/bash
# Install the repository-pinned training addition to an isolated LUMI overlay.

set -euo pipefail

CONTAINER=${OELLM_CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}
: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
OVERLAY=${OELLM_PYTHON_OVERLAY:-$OELLM_RUN_ROOT/python/trl-1.4.0}
BIND=/pfs,/scratch

if singularity exec -B "$BIND" "$CONTAINER" env PYTHONPATH="$OVERLAY" \
  python3 -c 'from importlib.metadata import version; from trl import SFTConfig; assert version("trl") == "1.4.0"; assert SFTConfig(output_dir="/tmp", loss_type="chunked_nll").loss_type == "chunked_nll"'; then
  echo "TRL 1.4.0 with chunked NLL is already installed in $OVERLAY"
  exit 0
fi

mkdir -p "$OVERLAY"
singularity exec -B "$BIND" "$CONTAINER" python3 -m pip install \
  --no-deps \
  --upgrade \
  --require-hashes \
  --target "$OVERLAY" \
  --requirement requirements-lumi.txt

singularity exec -B "$BIND" "$CONTAINER" env PYTHONPATH="$OVERLAY" \
  python3 -c 'from importlib.metadata import version; from trl import SFTConfig; assert version("trl") == "1.4.0"; assert SFTConfig(output_dir="/tmp", loss_type="chunked_nll").loss_type == "chunked_nll"'
