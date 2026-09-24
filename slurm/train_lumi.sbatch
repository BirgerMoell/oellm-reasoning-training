#!/bin/bash
#SBATCH --job-name=oellm-reason-sft
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
module purge

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${TRAIN_CONFIG:=configs/train/smoke.yaml}"
: "${GPUS_PER_NODE:=8}"
: "${RESUME_FROM_CHECKPOINT:=0}"
: "${CHAT_TEMPLATE_FILE:=templates/oellm_gemma_assistant_mask.jinja}"

if [[ "$CHAT_TEMPLATE_FILE" != /* ]]; then
  CHAT_TEMPLATE_FILE=$SLURM_SUBMIT_DIR/$CHAT_TEMPLATE_FILE
fi
if [[ ! -s "$CHAT_TEMPLATE_FILE" ]]; then
  echo "Training preflight failed: missing chat template $CHAT_TEMPLATE_FILE" >&2
  exit 2
fi

PRODUCTION_CONFIG=configs/train/reasoning-v1.yaml
PRODUCTION_DATA=$OELLM_RUN_ROOT/data/reasoning-v1/train.parquet
PRODUCTION_MANIFEST=$OELLM_RUN_ROOT/data/reasoning-v1/manifest.json
PRODUCTION_OUTPUT=$OELLM_RUN_ROOT/checkpoints/reasoning-v1
RUN_DATA=$(awk '/^[[:space:]]+train: / {print $2; exit}' "$TRAIN_CONFIG")
if [[ -z "$RUN_DATA" ]]; then
  echo "Training preflight failed: could not resolve train parquet from $TRAIN_CONFIG" >&2
  exit 2
fi
RUN_MANIFEST=$(dirname "$RUN_DATA")/manifest.json

if [[ "$TRAIN_CONFIG" == "$PRODUCTION_CONFIG" ]]; then
  for required in "$PRODUCTION_DATA" "$PRODUCTION_MANIFEST"; do
    if [[ ! -s "$required" ]]; then
      echo "Production preflight failed: missing or empty $required" >&2
      exit 2
    fi
  done
  if [[ "$RESUME_FROM_CHECKPOINT" == "0" ]] && \
     [[ -d "$PRODUCTION_OUTPUT" ]] && \
     [[ -n "$(find "$PRODUCTION_OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Production preflight failed: $PRODUCTION_OUTPUT is not empty; set RESUME_FROM_CHECKPOINT explicitly" >&2
    exit 2
  fi
  if [[ "$RESUME_FROM_CHECKPOINT" != "0" ]] && [[ ! -d "$PRODUCTION_OUTPUT" ]]; then
    echo "Production preflight failed: resume requested but $PRODUCTION_OUTPUT does not exist" >&2
    exit 2
  fi
fi

CONTAINER=${OELLM_CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}
OVERLAY=${OELLM_PYTHON_OVERLAY:-$OELLM_RUN_ROOT/python/trl-1.4.0}
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
HF_CACHE=$OELLM_RUN_ROOT/cache/huggingface

if [[ ! -d "$OVERLAY/trl" ]]; then
  echo "Training preflight failed: isolated TRL 1.4.0 overlay is missing: $OVERLAY" >&2
  echo "Run scripts/install_lumi_dependencies.sh on a login node first." >&2
  exit 2
fi

mkdir -p logs "$HF_CACHE" "$OELLM_RUN_ROOT/runs"
ln -sf "${SLURM_JOB_ID}.out" logs/latest.out 2>/dev/null || true

RUN_DIR=$OELLM_RUN_ROOT/runs/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
export RUN_DIR
cp "$TRAIN_CONFIG" "$RUN_DIR/resolved-train-config.yaml"
cp "$CHAT_TEMPLATE_FILE" "$RUN_DIR/chat-template.jinja"
if [[ -s "$RUN_MANIFEST" ]]; then
  cp "$RUN_MANIFEST" "$RUN_DIR/data-manifest.json"
fi
git rev-parse HEAD > "$RUN_DIR/repository-git-sha.txt"
git status --porcelain=v1 > "$RUN_DIR/repository-status.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$RUN_DIR/slurm-job.txt"
sha256sum "$RUN_DIR/resolved-train-config.yaml" "$RUN_DIR/chat-template.jinja" \
  > "$RUN_DIR/provenance-sha256.txt"
if [[ -s "$RUN_DIR/data-manifest.json" ]]; then
  sha256sum "$RUN_DIR/data-manifest.json" >> "$RUN_DIR/provenance-sha256.txt"
fi
export REPOSITORY_GIT_SHA
REPOSITORY_GIT_SHA=$(git rev-parse HEAD)
export TRAINING_CONTAINER=$CONTAINER

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET
export NCCL_DEBUG_FILE="$RUN_DIR/rccl.%h.%p.log"
export OMP_NUM_THREADS=7
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$RUN_DIR/torch-nccl-trace-"
export RANK0_META_LOAD=1
export RESUME_FROM_CHECKPOINT

NUM_PROC=$(( GPUS_PER_NODE * SLURM_NNODES ))
export NUM_PROC
echo "Config=$TRAIN_CONFIG nodes=$SLURM_NNODES world=$NUM_PROC root=$OELLM_RUN_ROOT"

set +e
srun --label singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  export HF_HOME=$HF_CACHE HF_DATASETS_CACHE=$HF_CACHE/datasets
  export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=1
  export RANK0_META_LOAD=$RANK0_META_LOAD RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT
  export CHAT_TEMPLATE_FILE=$CHAT_TEMPLATE_FILE
  export TRAIN_CONFIG=$TRAIN_CONFIG
  export OELLM_RUN_ROOT=$OELLM_RUN_ROOT REPOSITORY_GIT_SHA=$REPOSITORY_GIT_SHA
  export TRAINING_CONTAINER=$TRAINING_CONTAINER
  export RUN_DIR=$RUN_DIR
  cd $SLURM_SUBMIT_DIR
  if [[ \$SLURM_PROCID == 0 ]]; then
    python3 -c \"import platform; import accelerate, datasets, torch, transformers, trl; assert trl.__version__ == '1.4.0'; print('platform=' + platform.platform()); print('python=' + platform.python_version()); print('torch=' + torch.__version__); print('transformers=' + transformers.__version__); print('trl=' + trl.__version__); print('accelerate=' + accelerate.__version__); print('datasets=' + datasets.__version__); print('loss_type=chunked_nll')\" > $RUN_DIR/environment.txt
    python3 scripts/write_run_record.py --run-dir $RUN_DIR --config $TRAIN_CONFIG --root $OELLM_RUN_ROOT --container $TRAINING_CONTAINER --status started
  fi
  accelerate launch \
    --config_file configs/accelerate/fsdp_lumi.yaml \
    --main_process_ip '$MASTER_ADDR' \
    --main_process_port '$MASTER_PORT' \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_PROC \
    --machine_rank \$SLURM_PROCID \
    scripts/train_sft.py --config $TRAIN_CONFIG
  if [[ \$SLURM_PROCID == 0 ]]; then
    python3 scripts/write_run_record.py --run-dir $RUN_DIR --config $TRAIN_CONFIG --root $OELLM_RUN_ROOT --container $TRAINING_CONTAINER --status completed
  fi
"
TRAIN_EXIT=$?
set -e

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
  singularity exec -B "$BIND" "$CONTAINER" bash -lc "
    set -euo pipefail
    export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
    export NUM_PROC=$NUM_PROC RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT
    cd $SLURM_SUBMIT_DIR
    python3 scripts/write_run_record.py --run-dir $RUN_DIR --config $TRAIN_CONFIG \
      --root $OELLM_RUN_ROOT --container $TRAINING_CONTAINER --status failed \
      --exit-code $TRAIN_EXIT
  " || true
  echo "Training job failed with exit code $TRAIN_EXIT at $(date --iso-8601=seconds)" >&2
  exit "$TRAIN_EXIT"
fi

echo "Training job completed at $(date --iso-8601=seconds)"
