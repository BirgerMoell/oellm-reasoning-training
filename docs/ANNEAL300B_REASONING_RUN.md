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
| Attention / loss | FlashAttention 2; Liger fused linear cross-entropy |
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

# GPU integration gate: same model/data/template/topology/optimizer as production, ten updates only.
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/train_anneal300b_sanity_lumi.sbatch

# Require finite logs and an exhaustive checkpoint scan before production.
sbatch --export=ALL,CHECKPOINT="$OELLM_RUN_ROOT/checkpoints/reasoning-anneal300b-sanity/checkpoint-10",FULL_SCAN=1 \
  slurm/check_checkpoint_lumi.sbatch

# Submit only with an afterok dependency on the successful weight audit.
sbatch --dependency=afterok:WEIGHT_AUDIT_JOB --kill-on-invalid-dep=yes \
  --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" \
  slurm/train_anneal300b_production_lumi.sbatch
```

The production job is not accepted merely because Slurm reports `COMPLETED`. All logged losses and
gradient norms must be finite, the final state must pass an exhaustive value scan, and the selected model
must pass comparative evaluation.

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
