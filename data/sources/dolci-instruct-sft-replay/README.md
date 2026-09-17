# Dolci Instruct SFT replay

| Field | Value |
|---|---|
| State | pinned public input for `reasoning-anneal300b-v1` |
| Allocation | 35% of the weighted rendered-token budget |
| Role | preserve instruction following, dialogue behavior, multilingual coverage, and the parent model's empty-think convention |
| Hugging Face | [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) |
| Revision | `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221` |
| Split / rows | `train`; 2,152,112 conversations |
| Format | Parquet `messages`; normalized to `role` and `content` only |
| License | ODC-By-1.0; research and educational use under Ai2's Responsible Use Guidelines |

This is the public source mixture used to train the requested parent checkpoint. The parent model card
records the dataset as `main` rather than a commit, so this repository pins the resolved public revision
visible when this follow-on recipe was created. That is reproducible, but it must not be described as a
cryptographic proof that the upstream dataset was byte-identical on the date of the parent run.

Replay rows are not required to contain reasoning. Under the native Qwen3 template, a normal final
assistant response is serialized with an empty `<think>...</think>` block, matching the parent SFT.
Reasoning rows from the other sources must instead contain both a non-empty thought and a final answer.
