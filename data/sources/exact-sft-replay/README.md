# Exact SFT mixture replay

| Field | Value |
|---|---|
| State | `production`, already materialized on LUMI |
| v1 allocation | 15% of rendered tokens |
| Role | preserve instruction following, dialogue style, safety behavior, and multilingual coverage |
| LUMI artifact | `/scratch/project_465002530/users/bmoell/posttrain-data/qwen35-9b-sft-parquet/train.parquet` |
| Verified rows | 1,082,196 |
| Format | Parquet; `id`, `messages` |
| Lineage | Tulu 3 plus EuroBlocks, as documented in the [starting model card](https://huggingface.co/birgermoell/oellm-9b-256k-sft) |
| License | composite; use the source model card and upstream manifests |

This is deliberately the exact artifact used in the prior SFT run rather than a newly assembled “similar”
instruction mix. Replay records are not required to contain explicit reasoning. They are trained with the
same assistant-only mask as reasoning records.

The builder verifies the 1,082,196-row count before use. A changed parquet requires a new data-recipe
version and manifest.
