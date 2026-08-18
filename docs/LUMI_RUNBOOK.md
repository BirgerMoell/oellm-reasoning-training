# LUMI runbook

## Fixed environment

| Resource | Path/version |
|---|---|
| Project | `project_465002530` |
| GPU partition | `standard-g` |
| Container | `/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif` |
| Python overlay | `/scratch/project_465002530/users/bmoell/pylibs-overlay` |
| PyTorch | `2.9.1+rocm6.4` |
| Transformers | `5.12.1` |
| TRL | `0.28.0` |
| Datasets | `5.0.0` |
| Accelerate | `1.12.0` |
| Liger Kernel | `0.8.1` (fused linear cross-entropy only) |
| Default artifact root | `/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts` |

Compute nodes run offline. Download model/data snapshots on a login node before the CPU build or GPU job.
All caches belong on scratch, never in the small LUMI home directory.

## 1. Clone and set the run root

```bash
cd /scratch/project_465002530/users/bmoell
git clone https://github.com/BirgerMoell/oellm-reasoning-training.git
cd oellm-reasoning-training
export OELLM_RUN_ROOT=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts
mkdir -p "$OELLM_RUN_ROOT" logs
```

Record `git rev-parse HEAD` before doing anything else.

Install and verify the repository-pinned addition to the shared overlay from the login node. The script
uses hashes from `requirements-lumi.txt`; compute nodes remain offline.

```bash
scripts/install_lumi_dependencies.sh
```

## 2. Stage the pinned snapshots

```bash
export HF_HOME="$OELLM_RUN_ROOT/cache/huggingface"
scripts/stage_lumi.sh
```

The command writes one `snapshot.json` per repository. Re-running is safe: the exact same revisions are
reused. A different revision requires editing the recipe and committing a new version.

Expected network materialization is large (Dolci and multilingual Nemotron contain many Parquet shards).
Use `--source <id>` to stage one source at a time. The exact SFT replay remains at its existing project path
and is never copied unless the run needs an immutable local replica.

## 3. Run the sampled integration sanity gate

```bash
SANITY_DATA_JOB=$(sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/build_data_sanity_lumi.sbatch | awk '{print $NF}')
echo "$SANITY_DATA_JOB"

# Submit only after the data job completed and validated its manifest.
SANITY_GPU_JOB=$(sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/train_sanity_lumi.sbatch | awk '{print $NF}')
echo "$SANITY_GPU_JOB"
```

This resolves every production glob but loads only its first pinned shard and at most 5,000 rows per slice,
building 16,777,216 rendered tokens. It then runs ten packed 16K updates on the production 8-node / 64-GCD
topology. The wrapper executes the exact production launcher; only data/output paths, step count, logging,
and save interval differ. Require finite loss on all ranks and a reloadable checkpoint. Never release or
resume production from `checkpoints/reasoning-sanity`.

The 16K workload requires Liger's Qwen3 fused linear cross-entropy. Without it, the ordinary loss
materializes a roughly 16 GiB full-vocabulary logits gradient per rank and fails after optimizer state is
created. Only the fused loss is enabled; attention, RoPE, RMSNorm, and SwiGLU remain on the pinned model
and FlashAttention implementations.

## 4. Build the mixture

```bash
BUILD_JOB=$(sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/build_data_lumi.sbatch | awk '{print $NF}')
echo "$BUILD_JOB"
```

Monitor `logs/<job>.out` and `logs/<job>.err`. The final lines report selected rows/tokens per slice and
the output SHA. The completed artifacts are:

```text
$OELLM_RUN_ROOT/data/reasoning-v1/train.parquet
$OELLM_RUN_ROOT/data/reasoning-v1/manifest.json
$OELLM_RUN_ROOT/data/reasoning-v1/dedup.sqlite3
```

Run the validator:

```bash
python3 scripts/validate_run.py --root "$OELLM_RUN_ROOT" \
  --config configs/data/reasoning-v1.yaml
```

## 5. Baseline evaluation

Materialize or link the baseline to `$OELLM_RUN_ROOT/models/oellm-9b-256k-sft`. Run the benchmark harness
with output root `$OELLM_RUN_ROOT/eval/baseline-08359ad/`. Keep raw generation JSONL, not only aggregate
scores. The detailed matrix is in `docs/EVALUATION.md`.

## 6. Full-artifact smoke

```bash
SMOKE_JOB=$(sbatch --nodes=1 --gpus-per-node=8 --time=0-02:00:00 \
  --export=ALL,GPUS_PER_NODE=8,OELLM_RUN_ROOT="$OELLM_RUN_ROOT",TRAIN_CONFIG=configs/train/smoke.yaml \
  slurm/train_lumi.sbatch | awk '{print $NF}')
echo "$SMOKE_JOB"
```

Accept only after checking finite loss, absence of packing/template warnings, checkpoint contents, and
the model invariant report. `COMPLETED` in Slurm is necessary but not sufficient.

## 7. Production

```bash
PROD_JOB=$(sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/train_production_lumi.sbatch | awk '{print $NF}')
echo "$PROD_JOB"
```

The production wrapper fixes the tested 8-node / 64-GCD topology, 14-hour wall time, and committed
`reasoning-v1` recipe. The common launcher refuses a fresh run if the output directory is non-empty and
refuses a resume if no production output exists. Based on the validated 64-GCD sanity throughput, expect
roughly 10–12 hours and 650–800 GCD-hours, including eight full-state checkpoint saves.

Record the job ID immediately in `$OELLM_RUN_ROOT/runs/<run-id>/run.yaml`. Watch the first 20 steps for
finite, generally decreasing loss and similar throughput on all nodes.

Useful commands:

```bash
squeue -j "$PROD_JOB"
sacct -j "$PROD_JOB" --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
tail -f "logs/${PROD_JOB}.out"
tail -f "logs/${PROD_JOB}.err"
```

## 8. Recovery

Production saves full trainer state every 250 steps. Set `RESUME_FROM_CHECKPOINT=1` to let Transformers
find the newest `checkpoint-*`, or pass an explicit checkpoint directory:

```bash
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT",RESUME_FROM_CHECKPOINT=1 \
  slurm/train_production_lumi.sbatch
```

Before resuming, verify that the checkpoint contains model, optimizer, scheduler, RNG, and trainer state.
If optimizer state is incomplete, treat it as a warm start with a new run ID and document the discontinuity.

## 9. Evaluate and export

Evaluate each candidate checkpoint from a separate batch job. Do not run long-context evaluation inside the
training allocation. When a checkpoint passes every gate, export it into
`$OELLM_RUN_ROOT/releases/<model-name>/`, validate architecture/tokenizer invariants, and upload from the
login node with a token supplied through the environment. Never commit credentials or put them in Slurm
files.

## LUMI-specific invariants

- Keep `RANK0_META_LOAD=1`; otherwise every rank may load a full 9B CPU copy and exhaust node RAM.
- Use one Slurm task per node; Accelerate spawns the eight local workers.
- Set `GPUS_PER_NODE=8` explicitly because `SLURM_GPUS_ON_NODE` can report physical GCDs rather than the
  requested count.
- Keep Hugging Face caches on scratch.
- Keep `flash_attention_2` with packing; other attention implementations can allow packed examples to
  cross-attend.
- Use `<start_of_turn>` / `<end_of_turn>`, never Qwen ChatML tokens.
- Keep `max_position_embeddings=262144` and `rope_theta=64000000` through every save/export.
