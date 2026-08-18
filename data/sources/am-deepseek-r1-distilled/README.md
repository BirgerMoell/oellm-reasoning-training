# AM DeepSeek-R1-0528 distilled collection

| Field | Value |
|---|---|
| State | `available-alternative`, not in production v1 |
| Role | large English math, code, science, instruction, and multiturn reasoning collection |
| LUMI SFT format | `/scratch/project_462000963/datasets/posttraining_data/SFTTrainer_format/eng/AM-DeepSeek-R1-0528-Distilled-think/` |
| LUMI Megatron format | `/scratch/project_462000963/datasets/posttraining_data/Megatron_format/am-deepseek-r1-think/` |
| Size | about 40 GB JSONL; about 44.9 GB Megatron `.bin` |
| Public revision | not yet pinned in this repository |
| License | must be resolved per exact upstream release before production use |

The collection is already separated into `math`, `code`, `science`, `if`, `multiturn`, and `other`.
Sample rows contain a flattened Llama-style template with explicit `<think>` traces. It is valuable for
an ablation against Dolci/Nemotron, but v1 excludes it so the production mix has public, immutable,
decontaminated revisions and the correct Gemma template is applied from structured messages.

To promote it, add a public upstream/revision, normalize to structured `messages`, decontaminate against
the evaluation suite, measure exact token/language distributions, and create a new recipe version.
