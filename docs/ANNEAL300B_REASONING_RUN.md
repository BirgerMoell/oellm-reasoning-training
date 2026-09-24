# Anneal-300B SFT reasoning continuation

## Target and objective

This run adds explicit, non-empty reasoning supervision to
[`Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft)
at immutable revision `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`.

The parent is a full-parameter 9B SFT checkpoint trained for about 2.8B tokens on all 2,152,112
Dolci-Instruct-SFT conversations. It already uses the desired Qwen3 ChatML format, but almost every
assistant response was rendered with an empty `<think>...</think>` block. This continuation teaches the
same model to place an actual trace inside that block and a user-facing answer after `</think>`.

This is reasoning SFT, not RLVR, DPO, or a claim of improved reasoning before evaluation.

## Why this differs from reasoning-v1

The earlier 2.097B-token continuation improved some tasks but regressed IFEval, GSM8K, and MMLU and
produced repetition loops under greedy decoding. Repeating that recipe on a new parent would not be a
controlled improvement. This run changes four variables intentionally:

| Control | Anneal-300B decision | Reason |
|---|---:|---|
| Total packed tokens | 524,288,000 | one quarter of the earlier continuation; limits catastrophic forgetting while preserving useful checkpoint milestones |
| Peak learning rate | `1.5e-6` | half the earlier rate for an already trained instruction checkpoint |
| Instruction replay | 35% of weighted tokens | protects general instruction and multilingual behavior; earlier replay was about 15% |
| Reasoning format | non-empty `<think>` plus non-empty final answer | prevents “reasoning” rows with an empty thought and a long answer from teaching the wrong channel |
| Lexical loop gate | reject any 30-gram repeated at least 20 times | removes strict loop candidates before token allocation |

## Data recipe

The machine-readable recipe is
[`configs/data/reasoning-anneal300b-v1.yaml`](../configs/data/reasoning-anneal300b-v1.yaml).
Every Hugging Face input and the parent model are pinned to 40-character revisions. Shares below apply
to the weighted budget after the fixed multilingual pilot has been consumed once.

| Source | Weighted share | Use | Required gates |
|---|---:|---|---|
| OpenEuroLLM multilingual reasoning pilot | all accepted 16K-eligible rows once | coverage floor across 37 languages | accepted, complete non-empty think, final answer, no strict loop |
| Dolci Think SFT 32B decontaminated | 22% | broad English reasoning | complete non-empty think, final answer, no strict loop |
| Dolci Think SFT 7B decontaminated | 15% | complementary English reasoning | complete non-empty think, final answer, no strict loop |
| Nemotron v2 multilingual de/fr/es/it | 5% each | high-volume European-language reasoning | complete non-empty think, final answer, no strict loop |
| OpenR1 Math 220K | 8% | verifier-backed math | complete verifier-positive generation, complete non-empty think, final answer, no strict loop |
| Dolci-Instruct-SFT replay | 35% | preserve the requested parent checkpoint's instruction distribution | valid final assistant response; reasoning is not required |

Rows are rendered and budgeted with the exact native ChatML template. The builder rejects whole examples
outside 64–16,384 rendered tokens instead of truncating away a final answer. It then deduplicates on a
language-scoped normalized user-prompt hash, selects quotas in rendered tokens, shuffles deterministically,
and writes a Parquet SHA-256 and full source/filter accounting to `manifest.json`.

The parent card records its Dolci dataset as `main`, not an immutable commit. This recipe therefore pins
the resolved public revision `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221` visible when the continuation
was created. It is the same named source and row count, but not cryptographic proof of the exact bytes used
by the parent run.

### Materialized training artifact

Build job `22211011` produced 328,413 rows and 524,316,945 rendered tokens. The Parquet SHA-256 is
`b8f23c0e709e0a0d72b0926ff09a13942229a0c17d690feb171fa70d6c9d4499`; the manifest SHA-256 is
`920b50ce5a72fa925e28383a037178da418cff4e64b2a89d8da5839a14bad9b1`. These are the actual selected
shares, which differ slightly from requested quotas because the fixed multilingual pilot is consumed
first and rows are selected whole rather than truncated.

| Materialized source | Rows | Rendered tokens | Actual share | Training role |
|---|---:|---:|---:|---|
| `Dolci-Think-SFT-32B-decontaminated` | 22,370 | 111,888,981 | 21.34% | broad English reasoning traces |
| `Dolci-Think-SFT-7B-decontaminated` | 15,789 | 76,289,125 | 14.55% | complementary English reasoning traces |
| `Dolci-Instruct-SFT` replay | 262,630 | 178,003,408 | 33.95% | preserve parent instruction behavior |
| Nemotron v2 decontaminated, German | 4,397 | 25,430,548 | 4.85% | German reasoning |
| Nemotron v2 decontaminated, French | 4,358 | 25,432,730 | 4.85% | French reasoning |
| Nemotron v2 decontaminated, Spanish | 4,430 | 25,430,836 | 4.85% | Spanish reasoning |
| Nemotron v2 decontaminated, Italian | 4,332 | 25,434,610 | 4.85% | Italian reasoning |
| `OpenR1-Math-220k`, verifier-positive | 6,756 | 40,697,838 | 7.76% | verified mathematical reasoning |
| `reasoning-traces-multilingual` pilot | 3,351 | 15,708,869 | 3.00% | coverage floor across 37 non-English languages |
| **Total** | **328,413** | **524,316,945** | **100%** | **66.05% reasoning / 33.95% instruction replay** |

`openeurollm/Dolci-Think-SFT-translated` is **not** an input to this artifact. That repository is a
separate machine-translated Dolci Think 32B release covering 12 languages, including Swedish. Using it
would be a new recipe/version and requires a fresh build, deduplication pass, sanity run, and provenance
manifest; it must not be silently substituted into this immutable run.

### Translated-Dolci v2 follow-up

The follow-up recipe performs exactly that new experiment under the distinct version
`reasoning-anneal300b-dolci-translated-v2`. It pins the translated dataset at
`ba4754ab30afb66e652c3690ef0390dcea4939cd` and allocates 2% of weighted tokens to each of Czech,
German, Greek, Spanish, Finnish, French, Italian, Dutch, Polish, Romanian, Swedish, and Ukrainian.
The 24% translated allocation replaces part of the v1 English Dolci and Nemotron allocation; 35%
instruction replay is unchanged. All filtering, masking, model, optimizer, sequence length, and topology
controls remain identical, so the resulting comparison isolates the data-mixture change as closely as
practical.

The v2 artifact must be rebuilt from source and pass its own 30-step/checkpoint scan gate. A v1 manifest
or checkpoint is not valid input to the v2 run.

## Format and loss invariants

- native turn start: `<|im_start|>` (token 3);
- assistant turn terminator: `<|im_end|>` (token 4);
- pretraining EOS remains `<eos>` (token 2), but it is not the SFT turn boundary;
- assistant-only loss covers the final assistant response and its `<|im_end|>` token;
- the native template parses `<think>trace</think>answer`, normalizes the thought block, and masks only
  the assistant span after the last user message;
- the runner fails before model loading if the configured turn terminator is absent from the tokenizer or
  absent from the supervised assistant mask.

The template in [`templates/oellm_qwen3_assistant_mask.jinja`](../templates/oellm_qwen3_assistant_mask.jinja)
is copied from the pinned parent checkpoint rather than translated from the older Gemma-style template.

## Training configuration

| Parameter | Value |
|---|---:|
| Method | full-parameter TRL SFT, FSDP |
| Allocation | 8 LUMI-G nodes / 64 MI250X GCDs |
| Sequence length | 16,384 |
| Global packed batch | 64 sequences, up to 1,048,576 tokens/update |
| Updates | 500 |
| Peak learning rate | `1.5e-6` |
| Warmup | 15 updates |
| Schedule | cosine |
| AdamW | β₁ 0.9, β₂ 0.95, ε `1e-8`, weight decay 0 |
| Gradient clipping | 1.0 |
| Precision | BF16 |
| Attention / loss | FlashAttention 2; TRL 1.4.0 chunked NLL; Liger disabled |
| Checkpoints | every 100 updates; retain all 5 |

Based on the measured earlier 64-GCD run, 500 updates plus five full-state saves should take roughly
3–4 hours wall time, or about 190–260 GCD-hours. The five-hour allocation ceiling is 320 GCD-hours.
Observed runtime and throughput replace this estimate once the run completes.

## Fail-closed execution order on LUMI

```bash
export OELLM_RUN_ROOT=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts
export DATA_CONFIG=configs/data/reasoning-anneal300b-v1.yaml

# Login node: stage the exact model and public Parquet snapshots.
scripts/stage_lumi.sh

# CPU node: materialize and fully validate the immutable training artifact.
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT",DATA_CONFIG="$DATA_CONFIG" \
  slurm/build_data_lumi.sbatch

# GPU recovery gate: same model/data/template/topology/optimizer as production, 30 updates.
# This deliberately crosses the step-15 boundary where job 22211014 stalled.
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/train_anneal300b_sanity_lumi.sbatch

# Require finite logs and an exhaustive checkpoint scan before production.
sbatch --export=ALL,CHECKPOINT="$OELLM_RUN_ROOT/checkpoints/reasoning-anneal300b-sanity-v2/checkpoint-30",FULL_SCAN=1 \
  slurm/check_checkpoint_lumi.sbatch

# Submit only with an afterok dependency on the successful weight audit.
sbatch --dependency=afterok:WEIGHT_AUDIT_JOB --kill-on-invalid-dep=yes \
  --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/train_anneal300b_production_lumi.sbatch
```

The production job is not accepted merely because Slurm reports `COMPLETED`. All logged losses and
gradient norms must be finite, the final state must pass an exhaustive value scan, and the selected model
must pass comparative evaluation.

## Job 22211014 incident and recovery

The first Anneal-300B production attempt completed 15 updates with finite metrics and normal throughput,
then stopped completing steps until Slurm killed it at the five-hour wall limit. The only printed metric
window, at step 10, was healthy: loss `0.8346`, gradient norm `0.4778`, token accuracy `0.7722`, and
10.47M processed tokens. An exhaustive pre-run scan had already shown all 9,101,947,904 parent values
were finite. The failure therefore does not look like malformed data, NaNs, or ordinary out-of-memory.

The stalled run used Liger 0.8.1 fused linear cross-entropy on ROCm with Triton 3.2.0; the runtime warned
that this Triton version was below its supported recommendation. All sampled GPUs remained at 100%
activity while no rank completed step 16, which is consistent with a GPU-kernel or RCCL collective stall.
There was no Python traceback, so the precise offending kernel cannot be proven from the surviving logs.

The recovery makes three bounded changes while leaving model, data, optimizer, schedule, precision,
packing, and topology unchanged:

1. use TRL 1.4.0's upstream pure-PyTorch `chunked_nll`, which avoids materializing the full
   sequence-by-vocabulary logits tensor without using the Liger fused kernel;
2. set the distributed timeout and ProcessGroup watchdog to 15 and 10 minutes respectively, enable the
   collective flight recorder/desynchronization diagnostics, and write per-rank RCCL logs;
3. persist per-node step-begin/step-end heartbeats and require a fresh 30-step gate with full-state saves
   at steps 10, 20, and 30 before another production allocation is submitted.

Passing step 30 and the exhaustive checkpoint scan validates the workaround for this failure boundary;
it does not retroactively prove Liger was the unique root cause.

## Checkpoint selection and evaluation

Evaluate the untouched parent and steps 100, 200, 300, 400, and 500 with identical prompts, templates,
seeds, stop tokens, and decoder settings. Select the earliest checkpoint that clears retention gates; do
not automatically select step 500.

Required tracks:

- reasoning: MATH-500, GSM8K/MGSM by language, ARC-Challenge, and a contamination-resistant held-out set;
- instruction retention: IFEval plus the repository multilingual development suite;
- knowledge retention: MMLU slices already used for reasoning-v1 comparison;
- code: execution-scored HumanEval/MBPP or the existing decontaminated code evaluation path;
- long context: the same retrieval lengths used for the 256K parent;
- repetition: greedy control and fixed-seed sampling, reporting strict-loop rate, max-token exhaustion,
  EOS completion, answer accuracy, response length, and semantic no-progress review;
- language behavior: answer-language match and Swedish/German/French/Spanish examples with visible traces.

An acceptable checkpoint must show a real reasoning gain without a material instruction, language,
knowledge, safety, or long-context regression. If no checkpoint clears the gates, retain the run as an
experimental artifact and do not publish it as an upgrade.
