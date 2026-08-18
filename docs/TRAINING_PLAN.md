# Training plan

## Objective

Turn the multilingual instruction checkpoint into a materially better reasoner without giving up the
properties already demonstrated by the SFT model: European-language instruction following, stable chat
formatting, and natural-word retrieval through roughly 262K context.

The starting point is exactly
[`birgermoell/oellm-9b-256k-sft@08359ad`](https://huggingface.co/birgermoell/oellm-9b-256k-sft/tree/08359ad61333263c067edaf290067fea5b103d34).
It is a dense 9.1B-parameter Qwen3 architecture using the OpenEuroLLM tokenizer and Gemma-style turn
markers. Its model card reports 63.2 on instruction following, 58.6 on grounded QA, but 5.9 on the
reasoning/math slice of the multilingual development suite. It is therefore a sensible checkpoint for a
dedicated reasoning continuation, not a reason to repeat general SFT from the base model.

## What this stage changes

- increase exposure to complete, high-quality reasoning traces in math, code, and STEM;
- consume every 16K-eligible accepted v0.2 pilot translation once across 37 non-English languages, then
  provide additional high-volume reasoning in German, French, Spanish, and Italian;
- retain 15% of the exact prior SFT data to reduce instruction/language forgetting;
- train only assistant tokens while preserving the original user/assistant serialization;
- increase training sequence length from 4K to 16K so long reasoning traces keep their final answers;
- leave the model’s 262,144-position configuration and RoPE theta untouched.

This stage does not include DPO/SimPO, GRPO/RLVR, tools, or long-context extension. Those require different
data/rewards and should be compared from the accepted reasoning checkpoint rather than mixed into one
hard-to-interpret job.

## Why this mixture

The core sources are public and immutable. The large Dolci and Nemotron sources are decontaminated against
MATH-500, AIME, AMC, JEEBench, GPQA, LiveCodeBench, HumanEval, MBPP, IFEval, AlpacaEval, and Arena-Hard.
Dolci contributes breadth; Nemotron contributes controllable domain and language splits; OpenR1 contributes
verifier-backed math; exact SFT replay protects capabilities the checkpoint already has. The multilingual
v0.2 pilot contributes a small 37-language coverage floor and is not treated as volume data.

The 2,097,152,000-token budget matches 2,000 fully packed updates at:

```
64 ranks × 1 sequence/rank × 16,384 tokens × 2,000 updates
```

The builder measures tokens after applying the actual model template. It never approximates the mixture
from row counts.

## Data construction

For each source, in recipe order:

1. Resolve the exact local snapshot and compare its recorded revision with the recipe.
2. Normalize to structured `messages`; no source-specific flattened template survives.
3. Require at least one user and one assistant message with non-empty text.
4. Apply source-specific correctness checks. Pilot translations must have `quality.accepted == true`;
   OpenR1 must have a positive verifier and a complete trace.
5. Render with the starting tokenizer and assistant-mask template.
6. Reject examples shorter than 64 tokens or longer than 16,384 tokens. Never truncate away an answer.
7. Compute the language-scoped normalized first-user-prompt SHA-256 and deduplicate globally. This keeps
   parallel translations while removing same-language duplicates. Higher-priority sources win a duplicate.
8. Select all 3,351 valid 16K-eligible multilingual-pilot rows once and record the 74 overlength rows as
   exclusions. Subtract its 15,725,624 rendered tokens from the 2.097B target, then deterministically
   allocate the remaining 2,081,426,376 tokens by weighted source quotas with seed `20260818`.
9. Combine and deterministically shuffle all selected slices.
10. Write `train.parquet` plus `manifest.json` with source revisions, input files, selected row/token counts,
   filter counts, output SHA-256, tokenizer revision, recipe SHA-256, and build time.

The manifest is the identity of the dataset. A rerun that produces a different manifest is a new artifact,
even if the friendly name remains `reasoning-v1`.

## Stage 0 — baseline evaluation

Run the exact evaluation suite on the starting checkpoint before training. Store raw generations and
scores under `artifacts/eval/baseline-08359ad/`. Fixed prompts, seeds, generation parameters, evaluator
versions, and few-shot templates must be reused for the candidate.

This prevents a “win” produced by a changed harness. It also supplies exact baseline values for the
relative gates in [`EVALUATION.md`](EVALUATION.md).

## Integration sanity gate

Before materializing the 2.097B-token artifact, use `reasoning-sanity`: it resolves the same pinned input
globs as production but loads one shard and at most 5,000 raw rows per slice, budgeting only 16,777,216
rendered tokens. It is
not a statistical training mixture and its checkpoint is never publishable. Its purpose is to prove all 12
source globs, three adapters, tokenizer/template path, language-scoped deduplication, manifest validation,
64-rank cross-node FSDP initialization, assistant masking, ten 16K updates, and checkpoint saving.

The sanity gate passes only when `build_data_sanity_lumi.sbatch` validates the artifact and the eight-node
`configs/train/sanity.yaml` run writes a reloadable checkpoint with finite loss on all ranks. The dedicated
Slurm wrapper requests the same eight nodes and executes the exact production launcher. Do not start the
full data build until both results pass.

## Stage 1 — data materialization

Run `stage_hf.py` on a login node, then `build_data_lumi.sbatch` on CPU resources. The result must pass
`validate_run.py` before any GPU submission. The validation checks:

- weighted token shares sum to 1.0 and selected weighted shares are within 0.25 percentage points;
- the multilingual pilot contributes exactly 3,351 rows spanning exactly 37 languages and records exactly
  74 `too_long` exclusions;
- model revision and architecture invariants match;
- every record has valid alternating conversational roles and an assistant target;
- no complete conversation exceeds 16,384 rendered tokens;
- the final Parquet and manifest hashes match;
- no duplicate normalized prompt hash remains;
- every source meets its token quota.

## Stage 2 — one-node smoke

Use `configs/train/smoke.yaml`: 8 GCDs, 8K packing, 10 updates, the first 4,096 materialized rows.

The smoke is accepted when:

- all eight ranks initialize and train;
- assistant masking contains some but not all tokens;
- initial and final losses are finite;
- no cross-attention packing warning appears;
- a reloadable checkpoint is written with optimizer/trainer state;
- saved config still reports 262,144 positions, theta 64,000,000, vocab 263,168;
- a short Swedish prompt and a short math prompt produce non-empty, correctly terminated text.

## Stage 3 — production reasoning SFT

Use `configs/train/reasoning-v1.yaml` on 8 nodes / 64 GCDs.

| Parameter | Value | Reason |
|---|---:|---|
| max sequence length | 16,384 | retains long traces at a tractable activation cost |
| packed sequence batch | 64 | one sequence per GCD; approximately 1.049M tokens/update |
| updates | 2,000 | approximately one pass over the materialized token budget |
| peak learning rate | `3e-6` | conservative continuation from an already useful SFT model |
| warmup | 3% | stabilize the distribution change |
| schedule | cosine | smooth decay to a low final update size |
| precision | bf16 | matches the published checkpoint and MI250X path |
| attention | FlashAttention 2 | required for safe, efficient packing |
| loss | assistant tokens only | avoid learning prompt text as completion behavior |
| checkpoints | every 250 steps, keep 2 | supports recovery within scratch limits |
| evaluation during train | none | long generation suites run separately and reproducibly |

Expected compute is about twice the token work of the published 4K SFT run. Eight nodes are chosen to
keep the wall clock inside a normal LUMI allocation while retaining a simple data-parallel/FSDP layout.
Record observed tokens/sec and peak memory in the run record; those measurements replace the estimate for
the next version.

## Stage 4 — checkpoint selection

Evaluate checkpoints at steps 500, 1,000, 1,500, and 2,000. The final step is not automatically the best.
Choose the earliest checkpoint that clears all retention gates and gives the best geometric mean across
math, code, scientific reasoning, and multilingual reasoning. Early selection limits over-specialization.

If reasoning rises but language or long-context behavior regresses, do not publish the candidate under the
production model name. Keep it as an experiment and revise the replay/language allocation in `reasoning-v2`.

## Stage 5 — release artifact

For an accepted checkpoint:

1. consolidate/recast to bf16 if the FSDP save is float32;
2. copy tokenizer and the original chat template; keep `<end_of_turn>` as generation EOS;
3. run `validate_run.py --model <export>` and the full evaluation one last time;
4. write the model card with exact input SHA, data-manifest SHA, config, LUMI job IDs, GPU-hours, and results;
5. upload to a versioned Hugging Face repository; do not move an existing tag silently;
6. store the release manifest and raw evaluation outputs in Git or object storage, linked from the card.

## Follow-on stages

Only after accepting reasoning SFT:

1. **RLVR/GRPO:** verifiable math, code, and structured tasks with no hidden-answer leakage.
2. **Instruction preference optimization:** multilingual helpfulness and safety pairs, compared against
   the reasoning-SFT checkpoint.
3. **Tool/agent training:** schema-valid tool calls, execution success, and multilingual instructions.
4. **Long-context repair:** only if 64K–256K evaluation shows an actual regression; use a separately
   versioned mixed continuation rather than hiding repair data inside reasoning-v1.
