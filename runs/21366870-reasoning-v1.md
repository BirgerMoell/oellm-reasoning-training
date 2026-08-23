# Reasoning-v1 production training — LUMI job 21366870

Result: **training completed**. Checkpoint selection and release evaluation are recorded separately
because a completed optimization job is not, by itself, an accepted production model.

## Scope and provenance

| Field | Value |
|---|---|
| Training start | 2026-08-19 16:58:02 EEST |
| Training end | 2026-08-20 03:34:14 EEST |
| Repository commit used by training | `f5621b4d7dd211499693dea918339ac92ccfd438` |
| Parent model | `birgermoell/oellm-9b-256k-sft@08359ad61333263c067edaf290067fea5b103d34` |
| Data-manifest SHA-256 | `f92330319af5f917b6d9a01898d2a5dfc8f322b6d4250db1ce2111baafe077d1` |
| Training-Parquet SHA-256 | `7cbb6ba4a69f457b50ffc89a124f20335719633c32e4ef47a3a844bbf48407ff` |
| Training-config SHA-256 | `041eed072c4b2b7a1e4b431dd17e9f93e346bc67d87ef56cc2cc7df274c2abbf` |
| Chat-template SHA-256 | `2d19351bf03093ba670493496b8a1cca5c53218983980dd42dc291e0caab3ae7` |
| Slurm job | `21366870`, `COMPLETED`, exit `0:0` |
| Allocation | 8 LUMI-G nodes, 64 AMD MI250X GCDs, world size 64 |
| Wall time | 10:36:12 |
| Allocation cost | 678.61 GCD-hours |
| Container | `laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif` |

## Materialized data

The deterministic builder selected 1,130,994 unique conversations and 2,097,196,255 rendered tokens.
It rejected records outside 64–16,384 tokens and deduplicated language-scoped normalized user prompts.
Shares below are observed shares in the immutable manifest, not requested nominal weights.

| Source slice | Rows | Rendered tokens | Share | Terms recorded by manifest |
|---|---:|---:|---:|---|
| OpenEuroLLM multilingual reasoning traces v0.2 pilot | 3,351 | 15,725,624 | 0.75% | CC-BY-4.0 |
| Nemotron v2 math, decontaminated | 91,910 | 72,850,226 | 3.47% | CC-BY-4.0 |
| Nemotron v2 code, decontaminated | 26,020 | 41,630,061 | 1.99% | CC-BY-4.0 |
| Nemotron v2 STEM, decontaminated | 281,012 | 145,700,137 | 6.95% | CC-BY-4.0 |
| Nemotron v2 German, decontaminated | 17,730 | 104,077,800 | 4.96% | CC-BY-4.0 |
| Nemotron v2 French, decontaminated | 17,896 | 104,077,810 | 4.96% | CC-BY-4.0 |
| Nemotron v2 Spanish, decontaminated | 18,310 | 104,085,146 | 4.96% | CC-BY-4.0 |
| Nemotron v2 Italian, decontaminated | 17,734 | 104,071,448 | 4.96% | CC-BY-4.0 |
| OpenR1 Math 220K, verified | 17,343 | 104,071,331 | 4.96% | Apache-2.0 |
| Dolci Think 32B, decontaminated | 125,534 | 561,988,057 | 26.80% | ODC-By-1.0 |
| Dolci Think 7B, decontaminated | 116,995 | 426,703,655 | 20.35% | composite; see upstream |
| Exact prior-SFT mixture replay | 397,159 | 312,214,960 | 14.89% | composite; see parent card |

Every source revision is pinned in `configs/data/reasoning-v1.yaml`. The materialized Parquet is 6.83 GB.
Data terms are not replaced by the Apache-2.0 license on this repository or on released weights.

## Training procedure and observations

The run used full-parameter TRL SFT with FSDP across 64 ranks, one packed 16,384-token sequence per rank,
assistant-only loss, FlashAttention 2, Qwen3 fused linear cross-entropy, BF16 computation, and gradient
checkpointing. AdamW used peak learning rate `3e-6`, cosine decay, 3% warmup, zero weight decay, and no
gradient accumulation. Eight full-state checkpoints were retained every 250 optimizer steps.

| Metric | Observed value |
|---|---:|
| Optimizer steps | 2,000 |
| Packed input tokens reported by trainer | 2.086 billion |
| Effective epoch | 0.9945 |
| Trainer runtime | 37,170 seconds |
| Throughput | 3.444 packed sequences/s; 0.054 steps/s |
| Mean training loss | 0.8245 |
| First logged loss (step 10) | 0.9382 |
| Final logged loss (step 2,000) | 0.8172 |
| First → final mean token accuracy | 0.7418 → 0.7646 |

All logged loss and gradient values were finite; the run completed without an OOM. The consolidated
training checkpoint was initially saved as one 36.41 GB float32 safetensors file by the FSDP export path.
The release procedure explicitly converts the selected checkpoint to compact BF16 shards.

## Weight audit

CPU job `21426582` streamed every value in step 2,000 using bounded-memory safetensors slices and reported:

```text
FULL_FINITE files=1 tensors=399 values=9101947904 nonfinite=0
```

The selected BF16 export receives the same exhaustive scan plus a real GPU forward/generation test before
publication.

## Evaluation execution

Smoke array `21428734` passed on both the untouched parent and step 2,000. Full array `21443216` compares
the parent with steps 500, 1,000, 1,500, and 2,000 on eight deterministic, chat-templated tasks and stores
raw samples. It completed 36/40 cells; the four candidate MATH-500 cells exceeded the original 12-hour
limit and were resubmitted unchanged as array `21483191` with a 24-hour limit.

The partial results already show that all four candidates materially regress IFEval and do not improve
English GSM8K. Any released checkpoint from this run must therefore be labeled experimental rather than
as having passed the production-retention gates. The final score table, selection rule, and raw-result
hashes will be added after the MATH retries complete.
