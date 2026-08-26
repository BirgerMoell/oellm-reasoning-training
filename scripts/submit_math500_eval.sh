#!/bin/bash
# Submit a restart-safe distributed MATH-500 run on one LUMI-G node.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to an immutable Hugging Face checkpoint directory}"
: "${MODEL_ID:=$(basename "$MODEL_PATH")}"
: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${EVAL_ROOT:=$OELLM_RUN_ROOT/eval/math500-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${EVAL_PARTITION:=standard-g}"
: "${EVAL_TIME:=0-10:00:00}"
: "${MATH_NUM_PROCESSES:=8}"
: "${MATH_MAX_GEN_TOKS:=8192}"
: "${EVAL_LIMIT:=}"

if (( MATH_NUM_PROCESSES < 2 || MATH_NUM_PROCESSES > 8 )); then
  echo "MATH_NUM_PROCESSES must be between 2 and 8" >&2
  exit 2
fi
mkdir -p "$EVAL_ROOT/jobs" logs

JOB_ID=$(sbatch --parsable \
  --partition="$EVAL_PARTITION" \
  --time="$EVAL_TIME" \
  --gpus-per-node="$MATH_NUM_PROCESSES" \
  --export=ALL,MODEL_PATH="$MODEL_PATH",MODEL_ID="$MODEL_ID",OELLM_RUN_ROOT="$OELLM_RUN_ROOT",EVAL_ROOT="$EVAL_ROOT",MATH_NUM_PROCESSES="$MATH_NUM_PROCESSES",MATH_MAX_GEN_TOKS="$MATH_MAX_GEN_TOKS",EVAL_LIMIT="$EVAL_LIMIT" \
  slurm/eval_math500_distributed_lumi.sbatch)

cat > "$EVAL_ROOT/submission-math500-$JOB_ID.txt" <<EOF
job_id=$JOB_ID
model_id=$MODEL_ID
model_path=$MODEL_PATH
processes=$MATH_NUM_PROCESSES
limit=$EVAL_LIMIT
partition=$EVAL_PARTITION
time_limit=$EVAL_TIME
result_dir=$EVAL_ROOT/results/$MODEL_ID/hendrycks_math500
EOF
printf 'job_id=%s\neval_root=%s\nmodel_id=%s\nprocesses=%s\n' \
  "$JOB_ID" "$EVAL_ROOT" "$MODEL_ID" "$MATH_NUM_PROCESSES"
