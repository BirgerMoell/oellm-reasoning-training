#!/bin/bash
#SBATCH --job-name=oellm-card-examples
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/model-card-examples-%j.out
#SBATCH --error=logs/model-card-examples-%j.err

set -euo pipefail

: "${RELEASE_DIR:?Set RELEASE_DIR to the exported model directory}"
: "${EXAMPLES_CONFIG:=configs/eval/model-card-examples.yaml}"
: "${EXAMPLES_OUTPUT:=artifacts/eval/model-card-examples.json}"

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --rocm -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  cd $SLURM_SUBMIT_DIR
  python3 scripts/generate_model_card_examples.py \
    --model '$RELEASE_DIR' \
    --config '$EXAMPLES_CONFIG' \
    --output '$EXAMPLES_OUTPUT'
"
