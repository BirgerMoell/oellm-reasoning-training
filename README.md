# OpenEuroLLM 9B reasoning training

Production repository for continuing full-parameter reasoning SFT from
[`birgermoell/oellm-9b-256k-sft`](https://huggingface.co/birgermoell/oellm-9b-256k-sft)
on LUMI. It pins the starting checkpoint and data revisions, materializes a token-budgeted
mixture, launches the tested TRL/FSDP stack, and records every artifact needed to reproduce a run.

This repository owns one stage: **reasoning SFT after instruction SFT**. Preference optimization,
RLVR, tool use, and safety training should consume its accepted checkpoint as separate stages.

## Production plan at a glance

| Item | Decision |
|---|---|
| Starting model | `birgermoell/oellm-9b-256k-sft@08359ad61333263c067edaf290067fea5b103d34` |
| Why this stage | The published checkpoint is useful for multilingual instruction following and long-context retrieval, but its published reasoning/math aggregate is 5.9 |
| Method | Full-parameter, assistant-only reasoning SFT with packed sequences and FlashAttention 2 |
| Core mixture | `reasoning-v1`, exactly 2,097,152,000 rendered-token target before packing |
| Language allocation | 65% English reasoning, 20% European-language reasoning (`de`, `fr`, `es`, `it`), 15% multilingual/general replay |
| Sequence length | 16,384 tokens; records that do not fit are rejected rather than losing the final answer |
| Production allocation | 8 LUMI-G nodes / 64 MI250X GCDs, global sequence batch 64, 2,000 updates |
| Optimizer | AdamW, peak LR `3e-6`, cosine decay, 3% warmup, bf16, gradient checkpointing |
| Architecture invariants | 262,144 max positions, RoPE theta 64,000,000, vocab 263,168, Gemma-style turn markers |
| LUMI artifact root | `/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts` |
| Acceptance rule | Reasoning improves while multilingual instruction, code, safety, and 256K retrieval stay within the gates in [`docs/EVALUATION.md`](docs/EVALUATION.md) |

## Reasoning-v1 data

Shares are measured in **rendered tokens**, not rows. This matters because reasoning traces vary from
hundreds to tens of thousands of tokens.

| Slice | Token share | Language | State | Use |
|---|---:|---|---|---|
| OpenEuroLLM Dolci Think 32B, decontaminated | 20% | English | pinned; stage on LUMI | broad reasoning teacher traces |
| OpenEuroLLM Dolci Think 7B, decontaminated | 15% | English | pinned; stage on LUMI | complementary reasoning traces |
| OpenEuroLLM Nemotron v2 `math` | 10% | English | pinned; stage on LUMI | mathematical reasoning |
| OpenEuroLLM Nemotron v2 `code` | 8% | English | pinned; stage on LUMI | code reasoning |
| OpenEuroLLM Nemotron v2 `stem` | 7% | English | pinned; stage on LUMI | science and technical reasoning |
| OpenEuroLLM Nemotron v2 multilingual | 20% | `de`, `fr`, `es`, `it` (5% each) | pinned; stage on LUMI | reasoning in European languages |
| OpenR1 Math 220K, verified default split | 5% | English | already on LUMI; pinned upstream | correctness-filtered math |
| Exact SFT training mixture replay | 15% | multilingual | already on LUMI | preserve instruction/language behavior |

The machine-readable allocation is [`configs/data/reasoning-v1.yaml`](configs/data/reasoning-v1.yaml).
Every source has a human-readable card under [`data/sources/`](data/sources/), including exact revision,
license, input format, filters, public URL, LUMI location, and role in the run.

## Run on LUMI

From a LUMI login node:

```bash
git clone https://github.com/BirgerMoell/oellm-reasoning-training.git
cd oellm-reasoning-training
export OELLM_RUN_ROOT=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts

# Internet is available on the login node, not the compute nodes.
scripts/stage_lumi.sh

# Build the deterministic, token-budgeted parquet. Run this as a CPU/data job for the full mix.
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" slurm/build_data_lumi.sbatch

# Fail closed before spending a multi-node allocation.
python3 scripts/validate_run.py --root "$OELLM_RUN_ROOT" --config configs/data/reasoning-v1.yaml
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT",TRAIN_CONFIG=configs/train/smoke.yaml \
  slurm/train_lumi.sbatch

# After the smoke log has finite loss and a valid saved model:
sbatch --nodes=8 --gpus-per-node=8 --time=1-12:00:00 \
  --export=ALL,GPUS_PER_NODE=8,OELLM_RUN_ROOT="$OELLM_RUN_ROOT",TRAIN_CONFIG=configs/train/reasoning-v1.yaml \
  slurm/train_lumi.sbatch
```

See [`docs/LUMI_RUNBOOK.md`](docs/LUMI_RUNBOOK.md) for the cold-start procedure, monitoring,
checkpoint recovery, and exact output layout. Do not submit the production job until the data manifest
and one-node smoke gate both pass.

## Pipeline

1. **Stage immutable inputs.** Download the pinned model and dataset snapshots on a login node.
2. **Normalize and filter.** Keep complete user/assistant conversations, reject malformed or overlength
   traces, require verified OpenR1 solutions, and deduplicate by normalized prompt hash.
3. **Budget by tokens.** Select each slice to its allocation and write one shuffled Parquet plus a
   checksummed manifest.
4. **Smoke.** Run ten 8K updates on one node; verify finite loss, assistant masking, and architecture.
5. **Train.** Run 2,000 packed 16K updates on eight nodes with resumable checkpoints.
6. **Evaluate.** Compare the SFT baseline and reasoning candidate on the same prompts and decoding.
7. **Publish only an accepted checkpoint.** Preserve the input revision, data manifest, config, Slurm
   job IDs, logs, metrics, and output SHA in the run record.

## Repository map

| Path | Purpose |
|---|---|
| [`configs/data/`](configs/data/) | immutable source revisions, split filters, and token allocation |
| [`configs/train/`](configs/train/) | smoke and production hyperparameters |
| [`data/sources/`](data/sources/) | one stateful data card per source |
| [`scripts/stage_hf.py`](scripts/stage_hf.py) | snapshot the exact model and public datasets |
| [`scripts/build_mix.py`](scripts/build_mix.py) | normalize, validate, deduplicate, token-budget, and materialize |
| [`scripts/train_sft.py`](scripts/train_sft.py) | text-only TRL/FSDP training entry point |
| [`scripts/validate_run.py`](scripts/validate_run.py) | fail-closed model, manifest, and data checks |
| [`slurm/`](slurm/) | LUMI data and GPU jobs |
| [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) | stage rationale and detailed choices |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | benchmark matrix and acceptance gates |
| [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) | artifact lineage and run-record contract |

## Current verified LUMI assets

As of 2026-08-18:

- SFT output: `/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/oellm9b-256k-sft`
- exact SFT replay parquet: `/scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-sft-parquet/train.parquet`
  (1,082,196 rows)
- historical reasoning parquet: `/scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-reasoning-sft-parquet/train.parquet`
  (1,526,602 rows)
- shared raw Nemotron v2: `/scratch/project_462000963/datasets/posttraining_data/Nemotron-Post-Training-Dataset-v2`
- shared OpenR1 Math 220K: `/scratch/project_462000963/datasets/posttraining_data/OpenR1-Math-220k/default-train.jsonl`
- tested container: `/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif`

The historical reasoning parquet is documented for comparison, but it is **not** the production-v1
input: it was built from raw, non-decontaminated sources and its second 100K slice is English Nemotron
math, despite an earlier comment calling it Finnish.

## Licensing

Code in this repository is Apache-2.0. Dataset licenses remain those of their upstream sources. The data
manifest records them per source; this repository never re-licenses or commits the dataset artifacts.
