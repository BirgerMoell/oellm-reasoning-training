# Glaive code assistant v3

| Field | Value |
|---|---|
| State | `available-alternative`, not in production v1 |
| Role | code instruction replay or code-reasoning ablation |
| Public source | [`glaiveai/glaive-code-assistant-v3`](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3) |
| Revision | `31a2e16324e6712f212d4361a768fc49295becff` |
| License | Apache-2.0 |
| Shared LUMI file | `/scratch/project_462000963/datasets/posttraining_data/glaive-code-assistant-v3/train.jsonl` |
| Size | 1.92 GB JSONL |
| Format | `question`, `answer` |

The source is available and easy to normalize, but it is not inherently a verified chain-of-thought
dataset. V1 uses decontaminated Nemotron code instead. A future code-heavy recipe should add execution
tests, benchmark decontamination, language detection, and exact prompt deduplication before assigning it
a token share.
