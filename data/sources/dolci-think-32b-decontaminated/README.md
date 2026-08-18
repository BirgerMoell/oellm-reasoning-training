# Dolci Think SFT 32B — decontaminated

| Field | Value |
|---|---|
| State | `production` |
| v1 allocation | 23.5% of rendered tokens |
| Role | broad English reasoning traces from the larger teacher |
| Public source | [`openeurollm/Dolci-Think-SFT-32B-decontaminated`](https://huggingface.co/datasets/openeurollm/Dolci-Think-SFT-32B-decontaminated) |
| Revision | `286dde7da11a0fdc9d60a639d1089a2007d94f29` |
| Rows | 2,252,837 |
| Format | Parquet; `messages`, `id`, `source` |
| License | ODC-By-1.0 |
| LUMI raw location | `$OELLM_RUN_ROOT/raw/datasets/openeurollm--Dolci-Think-SFT-32B-decontaminated/` |

Use the OpenEuroLLM copy rather than the upstream raw copy because 847 rows matching the project’s
reasoning, code, instruction, and chat evaluation suite were removed. Preserve the conversation turns
and teacher response. Reject records that do not have a non-empty user prompt and assistant response,
or whose complete rendered conversation exceeds 16,384 tokens. Exact prompt deduplication is global,
so a prompt selected from this 32B source wins over the later 7B source.

The materialized manifest records selected rows and tokens; the snapshot itself stays immutable.
