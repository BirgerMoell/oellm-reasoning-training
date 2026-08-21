#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${EVAL_KIND:=full}"
: "${EVAL_MAX_CONCURRENT:=10}"
: "${EVAL_LIMIT:=}"
: "${EVAL_PARTITION:=small-g}"
: "${EVAL_TIME:=0-12:00:00}"

CONFIG=${1:-configs/eval/reasoning-v1.yaml}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EVAL_ROOT=${EVAL_ROOT:-$OELLM_RUN_ROOT/eval/reasoning-v1-$EVAL_KIND-$STAMP}
MATRIX=$EVAL_ROOT/matrix.tsv
MANIFEST=$EVAL_ROOT/matrix.manifest.json
CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
EVAL_CONTAINER=/pfs/lustrep4/scratch/project_462000963/oellm-cli-shared-evals/eval_env-lumi.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

mkdir -p "$EVAL_ROOT" logs
SELECTORS=()
if [[ "$EVAL_KIND" == "smoke" ]]; then
  SELECTORS+=(--model baseline --model step-2000 --task gsm8k)
elif [[ "$EVAL_KIND" != "full" ]]; then
  echo "EVAL_KIND must be smoke or full" >&2
  exit 2
fi

singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH=$OVERLAY:\${PYTHONPATH:-}
  cd $ROOT_DIR
  python3 scripts/build_eval_matrix.py \
    --config $CONFIG --root $OELLM_RUN_ROOT --output $MATRIX --manifest $MANIFEST \
    ${SELECTORS[*]}
"

ROWS=$(( $(wc -l < "$MATRIX") - 1 ))
if (( ROWS < 1 )); then
  echo "Evaluation matrix has no rows" >&2
  exit 2
fi

# Fail on the login node if a task or dataset is absent from the offline cache.
HF_HOME=$OELLM_RUN_ROOT/eval/hf-cache \
HF_DATASETS_CACHE=$OELLM_RUN_ROOT/eval/hf-cache/datasets \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONNOUSERSITE=1 PYTHONPATH=$OELLM_RUN_ROOT/eval/python \
singularity exec -B "$BIND" "$EVAL_CONTAINER" \
  python scripts/verify_eval_cache.py "$MATRIX"

ARRAY_END=$(( ROWS - 1 ))
JOB_ID=$(sbatch --parsable \
  --partition="$EVAL_PARTITION" \
  --time="$EVAL_TIME" \
  --array="0-${ARRAY_END}%${EVAL_MAX_CONCURRENT}" \
  --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT",EVAL_ROOT="$EVAL_ROOT",EVAL_MATRIX="$MATRIX",EVAL_LIMIT="$EVAL_LIMIT" \
  slurm/eval_lumi.sbatch)

cat > "$EVAL_ROOT/submission.txt" <<EOF
job_id=$JOB_ID
kind=$EVAL_KIND
rows=$ROWS
max_concurrent=$EVAL_MAX_CONCURRENT
limit=$EVAL_LIMIT
partition=$EVAL_PARTITION
time_limit=$EVAL_TIME
config=$CONFIG
matrix=$MATRIX
manifest=$MANIFEST
EOF
printf 'job_id=%s\neval_root=%s\nrows=%s\npartition=%s\ntime_limit=%s\n' \
  "$JOB_ID" "$EVAL_ROOT" "$ROWS" "$EVAL_PARTITION" "$EVAL_TIME"
