#!/bin/bash
#SBATCH --job-name=oellm-eval-packed
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval-packed-%j.out
#SBATCH --error=logs/eval-packed-%j.err

set -euo pipefail

: "${EVAL_ROWS:?Set EVAL_ROWS to comma-separated zero-based matrix rows}"
: "${EVAL_MATRIX:?Set EVAL_MATRIX to the generated TSV}"
: "${EVAL_ROOT:?Set EVAL_ROOT to the existing result root}"
: "${OELLM_RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts}"
: "${EVAL_LIMIT:=}"

IFS=',' read -ra ROWS <<< "$EVAL_ROWS"
if (( ${#ROWS[@]} < 1 || ${#ROWS[@]} > 4 )); then
  echo "Packed evaluation requires one to four rows; got ${#ROWS[@]}" >&2
  exit 2
fi

mkdir -p "$EVAL_ROOT/packed-logs" logs
pids=()
for row in "${ROWS[@]}"; do
  log_prefix=$EVAL_ROOT/packed-logs/${SLURM_JOB_ID}-row-${row}
  srun --exclusive --ntasks=1 --cpus-per-task=14 --gpus-per-task=1 --gpu-bind=single:1 \
    env EVAL_ROW_INDEX="$row" EVAL_JOB_TAG="${SLURM_JOB_ID}_packed_${row}" \
    scripts/run_eval_row_lumi.sh >"${log_prefix}.out" 2>"${log_prefix}.err" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "At least one packed evaluation failed; inspect $EVAL_ROOT/packed-logs" >&2
  exit 1
fi
