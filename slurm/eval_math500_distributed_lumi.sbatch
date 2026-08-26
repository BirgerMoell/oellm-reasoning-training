#!/bin/bash
# Distributed MATH-500 evaluation: one model replica per visible MI250X GCD.
# Submit through scripts/submit_math500_eval.sh so the run record is retained.
#SBATCH --job-name=oellm-math500
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0-10:00:00
#SBATCH --output=logs/math500-%j.out
#SBATCH --error=logs/math500-%j.err

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to an immutable Hugging Face checkpoint directory}"
: "${MODEL_ID:?Set MODEL_ID to a stable result identifier}"
: "${EVAL_ROOT:?Set EVAL_ROOT to an isolated evaluation directory}"
: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${MATH_NUM_PROCESSES:=8}"
: "${MATH_TASK:=hendrycks_math500}"
: "${MATH_NUM_FEWSHOT:=0}"
: "${MATH_MAX_GEN_TOKS:=8192}"
: "${EVAL_LIMIT:=}"

if (( MATH_NUM_PROCESSES < 2 || MATH_NUM_PROCESSES > 8 )); then
  echo "MATH_NUM_PROCESSES must be between 2 and 8; got $MATH_NUM_PROCESSES" >&2
  exit 2
fi
for required in "$MODEL_PATH/config.json" "$MODEL_PATH/tokenizer_config.json"; do
  [[ -s "$required" ]] || { echo "Missing model artifact: $required" >&2; exit 2; }
done
compgen -G "$MODEL_PATH/model*.safetensors" >/dev/null || {
  echo "No model safetensors found under $MODEL_PATH" >&2
  exit 2
}

CONTAINER=/pfs/lustrep4/scratch/project_462000963/oellm-cli-shared-evals/eval_env-lumi.sif
HF_HOME=$OELLM_RUN_ROOT/eval/hf-cache
HF_DATASETS_CACHE=$HF_HOME/datasets
EVAL_PYTHONPATH=$OELLM_RUN_ROOT/eval/python
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
RESULT_DIR=$EVAL_ROOT/results/$MODEL_ID/$MATH_TASK
JOB_JSON=$EVAL_ROOT/jobs/${SLURM_JOB_ID}_math500.json
LOCK_DIR=$RESULT_DIR/.running.lock

mkdir -p "$RESULT_DIR" "$EVAL_ROOT/jobs" logs
if find "$RESULT_DIR" -type f -name 'results_*.json' -print -quit | grep -q .; then
  echo "MATH-500 result already complete: $RESULT_DIR"
  exit 0
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another evaluation owns the result lock: $LOCK_DIR" >&2
  exit 2
fi

write_status() {
  local status=$1
  local exit_code=${2:-0}
  python3 - "$JOB_JSON" "$status" "$exit_code" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, status, exit_code = sys.argv[1:]
payload = {}
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
payload.update({
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "model_id": os.environ["MODEL_ID"],
    "model_path": os.environ["MODEL_PATH"],
    "task": os.environ.get("MATH_TASK", "hendrycks_math500"),
    "num_processes": int(os.environ.get("MATH_NUM_PROCESSES", "8")),
    "num_fewshot": int(os.environ.get("MATH_NUM_FEWSHOT", "0")),
    "max_gen_toks": int(os.environ.get("MATH_MAX_GEN_TOKS", "8192")),
    "limit": os.environ.get("EVAL_LIMIT", ""),
    "status": status,
    "exit_code": int(exit_code),
    "updated_at": datetime.now(timezone.utc).isoformat(),
})
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

finish() {
  local exit_code=$?
  if (( exit_code == 0 )); then
    write_status completed 0
  else
    write_status failed "$exit_code"
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap finish EXIT

export MODEL_PATH MODEL_ID MATH_TASK MATH_NUM_PROCESSES MATH_NUM_FEWSHOT MATH_MAX_GEN_TOKS EVAL_LIMIT
export HF_HOME HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONPATH=$EVAL_PYTHONPATH TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$(( 56 / MATH_NUM_PROCESSES ))
LIMIT_ARGS=()
[[ -z "$EVAL_LIMIT" ]] || LIMIT_ARGS+=(--limit "$EVAL_LIMIT")
MAIN_PROCESS_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

write_status running 0
echo "model=$MODEL_ID task=$MATH_TASK processes=$MATH_NUM_PROCESSES result_dir=$RESULT_DIR"
srun --ntasks=1 --cpus-per-task=56 --gpus-per-task="$MATH_NUM_PROCESSES" \
  singularity exec --rocm -B "$BIND" "$CONTAINER" env \
    HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PYTHONNOUSERSITE=1 PYTHONPATH="$EVAL_PYTHONPATH" TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    accelerate launch \
      --multi_gpu \
      --num_machines 1 \
      --num_processes "$MATH_NUM_PROCESSES" \
      --mixed_precision bf16 \
      --main_process_port "$MAIN_PROCESS_PORT" \
      -m lm_eval run \
      --model hf \
      --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True" \
      --tasks "$MATH_TASK" \
      --num_fewshot "$MATH_NUM_FEWSHOT" \
      --batch_size 1 \
      --gen_kwargs "do_sample=False,max_gen_toks=$MATH_MAX_GEN_TOKS" \
      --output_path "$RESULT_DIR" \
      --log_samples \
      --apply_chat_template \
      --trust_remote_code \
      --confirm_run_unsafe_code \
      --seed 20260821 \
      "${LIMIT_ARGS[@]}"

find "$RESULT_DIR" -type f -name 'results_*.json' -print -quit | grep -q . || {
  echo "Harness exited without a results JSON under $RESULT_DIR" >&2
  exit 1
}
