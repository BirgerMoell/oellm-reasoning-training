# Reasoning sanity — LUMI job 21348717

Result: **passed**. This is an integration artifact, not a publishable model and not a
production starting point.

## Scope and provenance

| Field | Value |
|---|---|
| Date | 2026-08-18 |
| Repository commit | `2502d9e1c6a9ea1b901fb0c1caefc5adf6a50d1e` |
| Parent model revision | `08359ad61333263c067edaf290067fea5b103d34` |
| Data build job | `21346805` (`COMPLETED`, 16m48s) |
| Training job | `21348717` (`COMPLETED`, exit `0:0`) |
| Exhaustive weight audit | `21351020` (`COMPLETED`, exit `0:0`, 4m18s) |
| Allocation | 8 nodes, 64 MI250X GCDs, 14m24s elapsed |
| Actual allocation cost | 15.36 GCD-hours |
| Config | `configs/train/sanity.yaml` |
| Data | `artifacts/data/reasoning-sanity/train.parquet` |
| Checkpoint | `artifacts/checkpoints/reasoning-sanity/checkpoint-10` |
| Final model | `artifacts/checkpoints/reasoning-sanity` |
| Container | `/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif` |

The sanity and production configs share the same model, 16,384-token sequence length, packing,
assistant-only loss, fused Qwen3 linear cross-entropy, FlashAttention 2, optimizer, precision,
learning-rate schedule, gradient checkpointing, and 64-rank launcher. Only the bounded dataset,
output path, step count, logging interval, and checkpoint interval differ.

## Data gate

- 10,198 unique conversations from all 12 production slices.
- 16,821,415 rendered tokens.
- Full prompt-hash deduplication scan passed.
- Parquet SHA-256:
  `7f93f6773311026ead4af20e73232615986f35e09c9f8afafb940cac82dd9a87`.
- Manifest SHA-256:
  `66a77205fc57bb6d932cb8be14fdb4a33c710b11c625ad4b9ad11bbf6bbf0939`.

## Training observations

| Metric | Observed value |
|---|---:|
| Optimizer steps | 10 / 10 |
| Trainer runtime | 429.5s |
| Trainer throughput | 1.49 packed sequences/s; 0.023 steps/s |
| Input tokens reported | 10,371,845 |
| Mean training loss | 0.8564 |
| Step 1 → step 10 loss | 0.8930 → 0.8200 |
| Observed gradient-norm range | 0.2874–0.4685 |
| Step 10 mean token accuracy | 0.7677 |

All 64 ranks formed the process group, every rank produced the expected non-empty assistant
mask (`8/22` tokens in the invariant probe), all ten losses and gradient norms were finite, and
the fused loss avoided the full-vocabulary FP32-logit allocation that the 16K path cannot afford.

## Saved state

The final artifact is 170 GiB and contains:

- full model and tokenizer at the output root;
- `checkpoint-10/model.safetensors` plus FSDP state;
- AdamW optimizer state, scheduler state, trainer state, and training arguments;
- one RNG state for each of the 64 ranks;
- the exact chat template and model/generation configs;
- `reasoning_training_metadata.json` with config, data-manifest, template, parent revision,
  container, repository SHA, Slurm job, node count, and world size.

The repository-native checker added in `aa67b56` streamed every saved value from
`checkpoint-10/model.safetensors` in bounded chunks on a 220 GiB CPU allocation. It reported
`FULL_FINITE files=1 tensors=399 values=9101947904 nonfinite=0`. This independently verifies
that the saved 9B model contains no NaN or infinity, rather than relying only on sampled weights
or finite training logs.

The production 2,000-step job was not submitted as part of this gate.
