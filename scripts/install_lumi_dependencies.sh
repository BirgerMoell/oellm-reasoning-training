#!/bin/bash
# Install the small, repository-pinned addition to the shared LUMI Python overlay.

set -euo pipefail

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch

if singularity exec -B "$BIND" "$CONTAINER" env PYTHONPATH="$OVERLAY" \
  python3 -c 'from importlib.metadata import version; assert version("liger-kernel") == "0.8.1"'; then
  echo "Liger Kernel 0.8.1 is already installed in $OVERLAY"
  exit 0
fi

singularity exec -B "$BIND" "$CONTAINER" python3 -m pip install \
  --no-deps \
  --require-hashes \
  --target "$OVERLAY" \
  --requirement requirements-lumi.txt

singularity exec -B "$BIND" "$CONTAINER" env PYTHONPATH="$OVERLAY" \
  python3 -c 'from importlib.metadata import version; assert version("liger-kernel") == "0.8.1"'
