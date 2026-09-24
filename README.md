# OpenEuroLLM 9B reasoning training

Production repository for continuing full-parameter reasoning SFT from
[`birgermoell/oellm-9b-256k-sft`](https://huggingface.co/birgermoell/oellm-9b-256k-sft)
on LUMI. It pins the starting checkpoint and data revisions, materializes a token-budgeted
mixture, launches the tested TRL/FSDP stack, and records every artifact needed to reproduce a run.

This repository owns one stage: **reasoning SFT after instruction SFT**. Preference optimization,
RLVR, tool use, and safety training should consume its accepted checkpoint as separate stages.

## Current Anneal-300B target

The repository now also contains a separate, format-native continuation for
[`Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft).
It does not overwrite or resume the published reasoning-v1 model.

| Field | Decision |
|---|---|
| Parent | pinned revision `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31` |
| Native format | Qwen3 ChatML; `<|im_end|>` is the supervised/serving stop token |
| Data | 524.3M rendered tokens; 65% reasoning and 35% parent-source Dolci instruction replay, plus the fixed multilingual pilot |
| Quality gates | non-empty `<think>` and final answer on every reasoning row; strict lexical loops rejected |
| Run | 500 packed 16K updates on 64 LUMI GCDs, peak LR `1.5e-6`, checkpoints every 100 steps |
| Safety gate | production-identical 30-step 64-GCD recovery run that crosses the previous step-15 stall, plus exhaustive weight scan before production |
| Current incident | job `22211014` timed out after step 15; the recovery switches from the Liger ROCm/Triton fused-loss path to TRL 1.4.0 `chunked_nll` and adds collective watchdogs |
| Detailed plan | [`docs/ANNEAL300B_REASONING_RUN.md`](docs/ANNEAL300B_REASONING_RUN.md) |

## Published checkpoint

| Field | Value |
|---|---|
| Model | [`birgermoell/oellm-9b-256k-reasoning-v1`](https://huggingface.co/birgermoell/oellm-9b-256k-reasoning-v1) |
| Checkpoint | reasoning-v1 step 2,000, unquantized BF16 |
| Training | LUMI job `21366870`, completed in 10:36:12 on 64 MI250X GCDs |
| Release validation | LUMI job `21492473`, passed GPU load/generation and an exhaustive scan of 9,101,947,904 values |
| Release status | **Experimental**: published for analysis and follow-on training; it did not pass the production-retention gates |
| Primary finding | ARC-Challenge and flexible multilingual MGSM improved, while GSM8K, IFEval, and MMLU college computer science regressed |
| Detailed provenance | [`runs/oellm-9b-256k-reasoning-v1-release.md`](runs/oellm-9b-256k-reasoning-v1-release.md) and the model card on Hugging Face |
| Repetition-loop analysis | [`docs/REPETITION_LOOPS.md`](docs/REPETITION_LOOPS.md): checkpoint-specific evidence, primary papers, audit code, and mitigation experiments |

## Production plan at a glance

| Item | Decision |
|---|---|
| Starting model | `birgermoell/oellm-9b-256k-sft@08359ad61333263c067edaf290067fea5b103d34` |
| Why this stage | The published checkpoint is useful for multilingual instruction following and long-context retrieval, but its published reasoning/math aggregate is 5.9 |
| Method | Full-parameter, assistant-only reasoning SFT with packed sequences, FlashAttention 2, and TRL chunked NLL |
| Core mixture | `reasoning-v1`, exactly 2,097,152,000 rendered-token target before packing |
| Language allocation | First include all 3,351 eligible pilot translations across 37 languages once; allocate the remaining tokens as 65% English reasoning, 20% `de/fr/es/it` reasoning, and 15% multilingual/general replay |
| Sequence length | 16,384 tokens; records that do not fit are rejected rather than losing the final answer |
| Production allocation | 8 LUMI-G nodes / 64 MI250X GCDs, global sequence batch 64, 2,000 updates |
| Measured budget | approximately 10–12 hours and 650–800 GCD-hours, including eight resumable checkpoints |
| Measured storage | 136 GiB per resumable checkpoint; reserve at least 1.2 TiB for eight checkpoints plus the final model |
| Optimizer | AdamW, peak LR `3e-6`, cosine decay, 3% warmup, bf16, gradient checkpointing |
| Architecture invariants | 262,144 max positions, RoPE theta 64,000,000, vocab 263,168, Gemma-style turn markers |
| LUMI artifact root | `/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts` |
| First execution gate | 16.78M-token sampled data build from all 12 slices, then 10 × 16K updates on the real 8-node / 64-GCD topology |
| Sanity gate status | Passed on 2026-08-18: data job `21346805`, GPU job `21348717`; see [`runs/21348717-reasoning-sanity.md`](runs/21348717-reasoning-sanity.md) |
| Acceptance rule | Reasoning improves while multilingual instruction, code, safety, and 256K retrieval stay within the gates in [`docs/EVALUATION.md`](docs/EVALUATION.md) |

## Reasoning-v1 data

Weighted shares are measured in **rendered tokens**, not rows. This matters because reasoning traces vary
from hundreds to tens of thousands of tokens. The multilingual pilot is a fixed coverage floor: consume
all 3,351 accepted rows that fit 16K once, record the 74 overlength exclusions, then allocate the remaining
token budget by weight.

| Slice | Token share | Language | State | Use |
|---|---:|---|---|---|
| OpenEuroLLM multilingual reasoning traces v0.2 pilot | all 3,351 16K-eligible rows once | 37 non-English languages | pinned public pilot | broad language coverage without truncating 74 overlength traces or oversampling 99 source problems |
| OpenEuroLLM Dolci Think 32B, decontaminated | 27% | English | pinned; broad high-capacity pool selected after specialized slices | broad reasoning teacher traces |
| OpenEuroLLM Dolci Think 7B, decontaminated | 20.5% | English | pinned; broad high-capacity pool selected after 32B Dolci | complementary reasoning traces |
| OpenEuroLLM Nemotron v2 `math` | 3.5% | English | pinned; capped below its measured 80.93M eligible-token capacity | mathematical reasoning without duplicate oversampling |
| OpenEuroLLM Nemotron v2 `code` | 2% | English | pinned; capped below its measured 56.07M unique-token capacity | code reasoning without duplicate oversampling |
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

# First prove every data adapter/path and the complete GPU stack on sampled data.
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" slurm/build_data_sanity_lumi.sbatch
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" slurm/train_sanity_lumi.sbatch

# Only after the sanity data manifest and ten-step checkpoint pass:
# Build the deterministic, token-budgeted parquet. Run this as a CPU/data job for the full mix.
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" slurm/build_data_lumi.sbatch

# Fail closed before spending a multi-node allocation.
python3 scripts/validate_run.py --root "$OELLM_RUN_ROOT" --config configs/data/reasoning-v1.yaml
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT",TRAIN_CONFIG=configs/train/smoke.yaml \
  slurm/train_lumi.sbatch

# After the smoke log has finite loss, exhaustively scan every saved weight:
SMOKE_CHECKPOINT="$OELLM_RUN_ROOT/checkpoints/reasoning-v1-smoke/checkpoint-10"
sbatch --export=ALL,CHECKPOINT="$SMOKE_CHECKPOINT",FULL_SCAN=1 \
  slurm/check_checkpoint_lumi.sbatch

# Submit production only after the exhaustive smoke checkpoint scan succeeds.
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/train_production_lumi.sbatch
```

For a long isolated full-data build, submit the same gates as a fail-closed dependency chain. The
production allocation can start only if the data job, atomic promotion, smoke run, and exhaustive
checkpoint scan all complete successfully:

```bash
scripts/submit_production_pipeline.sh \
  DATA_JOB_ID /scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts-final-GIT_SHA \
  FULL_DATA_BUILD_GIT_SHA
```

See [`docs/LUMI_RUNBOOK.md`](docs/LUMI_RUNBOOK.md) for the cold-start procedure, monitoring,
checkpoint recovery, and exact output layout. Do not submit the production job until the data manifest
and one-node smoke gate both pass.

## Pipeline

1. **Stage immutable inputs.** Download the pinned model and dataset snapshots on a login node.
2. **Sanity.** Resolve every production glob, load one pinned shard and at most 5,000 rows per slice, build
   16.78M tokens, and run ten packed 16K updates on the same 8-node / 64-rank launcher as production. This
   exercises every adapter, path, tokenizer, mask, inter-node FSDP rank, and save path.
3. **Normalize and filter.** Keep complete user/assistant conversations, require accepted pilot rows and
   verified OpenR1 solutions, reject malformed or overlength traces, and deduplicate by language-scoped
   normalized prompt hash.
4. **Budget by tokens.** Consume the fixed pilot floor, allocate the remaining tokens by source weight,
   and write one shuffled Parquet plus a checksummed manifest.
5. **Smoke.** Run ten 8K updates on the full artifact; verify finite loss, assistant masking, and architecture.
6. **Train.** Run 2,000 packed 16K updates on eight nodes, retaining all eight resumable
   250-step checkpoints so evaluation milestones are not deleted.
7. **Evaluate.** Compare the SFT baseline and reasoning candidate on the same prompts and decoding.
8. **Classify and publish deliberately.** Preserve the input revision, data manifest, config, Slurm
   job IDs, logs, metrics, and output SHA in the run record. A checkpoint that misses retention gates
   must be clearly labeled experimental rather than presented as a production upgrade.

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
| [`scripts/audit_source_capacity.py`](scripts/audit_source_capacity.py) | read-only eligible-token capacity audit for one or more recipe slices |
| [`scripts/audit_repetition.py`](scripts/audit_repetition.py) | exact-loop, short-repetition, and reasoning-tag audit for public outputs and the selected training mix |
| [`scripts/check_checkpoint_finite.py`](scripts/check_checkpoint_finite.py) | bounded-memory sampled or exhaustive safetensors finiteness audit |
| [`scripts/write_run_record.py`](scripts/write_run_record.py) | started/completed YAML provenance for every training attempt |
| [`slurm/`](slurm/) | LUMI data and GPU jobs |
| [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) | stage rationale and detailed choices |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | benchmark matrix and acceptance gates |
| [`docs/REPETITION_LOOPS.md`](docs/REPETITION_LOOPS.md) | evidence-backed diagnosis and mitigation plan for reasoning loops |
| [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) | artifact lineage and run-record contract |
| [`runs/`](runs/) | executed LUMI gate records and observed metrics |

## Current verified LUMI assets

As of 2026-08-18:

- SFT output: `/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/oellm9b-256k-sft`
- exact SFT replay parquet: `/scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-sft-parquet/train.parquet`
  (1,082,196 rows)
- historical reasoning parquet: `/scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-reasoning-sft-parquet/train.parquet`
  (1,526,602 rows)
- shared raw Nemotron v2: `/scratch/project_462000963/datasets/posttraining_data/Nemotron-Post-Training-Dataset-v2`
- shared OpenR1 Math 220K: `/scratch/project_462000963/datasets/posttraining_data/OpenR1-Math-220k/default-train.jsonl`
- tested container: `/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif` (all run scripts accept an `OELLM_CONTAINER` override)
- isolated training overlay: `trl==1.4.0`, hash-pinned in `requirements-lumi.txt`; its upstream
  `chunked_nll` computes the same token NLL in chunks so 16K × 263K-vocabulary training does not
  materialize full logits. The Liger ROCm/Triton loss path is disabled after job `22211014` stalled.

The historical reasoning parquet is documented for comparison, but it is **not** the production-v1
input: it was built from raw, non-decontaminated sources and its second 100K slice is English Nemotron
math, despite an earlier comment calling it Finnish.

## Licensing

Code in this repository is Apache-2.0. Dataset licenses remain those of their upstream sources. The data
manifest records them per source; this repository never re-licenses or commits the dataset artifacts.
