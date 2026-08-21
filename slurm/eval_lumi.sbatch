#!/bin/bash
#SBATCH --job-name=oellm-reason-eval
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/eval-%A_%a.out
#SBATCH --error=logs/eval-%A_%a.err

set -euo pipefail

: "${EVAL_MATRIX:?Set EVAL_MATRIX to the generated TSV}"
: "${EVAL_ROOT:?Set EVAL_ROOT to an isolated output directory}"
: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${EVAL_LIMIT:=}"

CONTAINER=/pfs/lustrep4/scratch/project_462000963/oellm-cli-shared-evals/eval_env-lumi.sif
HF_HOME=$OELLM_RUN_ROOT/eval/hf-cache
HF_DATASETS_CACHE=$HF_HOME/datasets
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

if [[ ! -s "$EVAL_MATRIX" ]]; then
  echo "Missing or empty evaluation matrix: $EVAL_MATRIX" >&2
  exit 2
fi

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$EVAL_MATRIX")
if [[ -z "$ROW" ]]; then
  echo "No matrix row for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=$'\t' read -r MODEL_ID MODEL_PATH TASK CAPABILITY NUM_FEWSHOT MAX_GEN_TOKS <<< "$ROW"

for required in "$MODEL_PATH/config.json" "$MODEL_PATH/tokenizer_config.json"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing model artifact: $required" >&2
    exit 2
  fi
done
if ! compgen -G "$MODEL_PATH/model*.safetensors" > /dev/null; then
  echo "No model safetensors found under $MODEL_PATH" >&2
  exit 2
fi

RESULT_DIR=$EVAL_ROOT/results/$MODEL_ID/$TASK
mkdir -p "$RESULT_DIR" "$EVAL_ROOT/jobs" logs
if find "$RESULT_DIR" -type f -name 'results_*.json' -print -quit | grep -q .; then
  echo "Refusing to overwrite completed result: $RESULT_DIR" >&2
  exit 2
fi

cat > "$EVAL_ROOT/jobs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json" <<EOF
{
  "array_job_id": "$SLURM_ARRAY_JOB_ID",
  "array_task_id": "$SLURM_ARRAY_TASK_ID",
  "model_id": "$MODEL_ID",
  "model_path": "$MODEL_PATH",
  "task": "$TASK",
  "capability": "$CAPABILITY",
  "num_fewshot": $NUM_FEWSHOT,
  "max_gen_toks": $MAX_GEN_TOKS,
  "limit": "${EVAL_LIMIT}",
  "status": "started"
}
EOF

export HF_HOME HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

LIMIT_ARGS=()
if [[ -n "$EVAL_LIMIT" ]]; then
  LIMIT_ARGS+=(--limit "$EVAL_LIMIT")
fi

echo "model=$MODEL_ID task=$TASK fewshot=$NUM_FEWSHOT max_gen_toks=$MAX_GEN_TOKS"
srun singularity exec --rocm -B "$BIND" "$CONTAINER" env \
  HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
  python -m lm_eval run \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASK" \
    --num_fewshot "$NUM_FEWSHOT" \
    --batch_size 1 \
    --device cuda:0 \
    --gen_kwargs "do_sample=False,max_gen_toks=$MAX_GEN_TOKS" \
    --output_path "$RESULT_DIR" \
    --log_samples \
    --apply_chat_template \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --seed 20260821 \
    "${LIMIT_ARGS[@]}"

python3 - "$EVAL_ROOT/jobs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json" <<'PY'
import json
import sys
from datetime import datetime, timezone

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
payload["status"] = "completed"
payload["completed_at"] = datetime.now(timezone.utc).isoformat()
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
