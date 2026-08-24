#!/bin/bash
# Run one zero-based row from an evaluation matrix inside an existing GPU step.
set -euo pipefail

: "${EVAL_MATRIX:?Set EVAL_MATRIX to the generated TSV}"
: "${EVAL_ROOT:?Set EVAL_ROOT to an isolated output directory}"
: "${EVAL_ROW_INDEX:?Set EVAL_ROW_INDEX to a zero-based matrix row}"
: "${EVAL_JOB_TAG:?Set EVAL_JOB_TAG to a unique Slurm/job identifier}"
: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${EVAL_LIMIT:=}"

CONTAINER=/pfs/lustrep4/scratch/project_462000963/oellm-cli-shared-evals/eval_env-lumi.sif
HF_HOME=$OELLM_RUN_ROOT/eval/hf-cache
HF_DATASETS_CACHE=$HF_HOME/datasets
EVAL_PYTHONPATH=$OELLM_RUN_ROOT/eval/python
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

ROW=$(sed -n "$((EVAL_ROW_INDEX + 2))p" "$EVAL_MATRIX")
if [[ -z "$ROW" ]]; then
  echo "No matrix row for index $EVAL_ROW_INDEX" >&2
  exit 2
fi
IFS=$'\t' read -r MODEL_ID MODEL_PATH TASK CAPABILITY NUM_FEWSHOT MAX_GEN_TOKS <<< "$ROW"

for required in "$MODEL_PATH/config.json" "$MODEL_PATH/tokenizer_config.json"; do
  [[ -s "$required" ]] || { echo "Missing model artifact: $required" >&2; exit 2; }
done
compgen -G "$MODEL_PATH/model*.safetensors" >/dev/null || {
  echo "No model safetensors found under $MODEL_PATH" >&2
  exit 2
}

RESULT_DIR=$EVAL_ROOT/results/$MODEL_ID/$TASK
mkdir -p "$RESULT_DIR" "$EVAL_ROOT/jobs" "$EVAL_ROOT/packed-logs"
if find "$RESULT_DIR" -type f -name 'results_*.json' -print -quit | grep -q .; then
  echo "Refusing to overwrite completed result: $RESULT_DIR" >&2
  exit 2
fi
LOCK_DIR=$RESULT_DIR/.running.lock
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another evaluation owns the result lock: $LOCK_DIR" >&2
  exit 2
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

JOB_JSON=$EVAL_ROOT/jobs/${EVAL_JOB_TAG}.json
cat > "$JOB_JSON" <<EOF
{
  "job_tag": "$EVAL_JOB_TAG",
  "matrix_row": $EVAL_ROW_INDEX,
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
export PYTHONNOUSERSITE=1 PYTHONPATH=$EVAL_PYTHONPATH TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-28}
LIMIT_ARGS=()
[[ -z "$EVAL_LIMIT" ]] || LIMIT_ARGS+=(--limit "$EVAL_LIMIT")

echo "row=$EVAL_ROW_INDEX model=$MODEL_ID task=$TASK fewshot=$NUM_FEWSHOT"
singularity exec --rocm -B "$BIND" "$CONTAINER" env \
  HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONNOUSERSITE=1 PYTHONPATH="$EVAL_PYTHONPATH" TOKENIZERS_PARALLELISM=false \
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

python3 - "$JOB_JSON" <<'PY'
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
