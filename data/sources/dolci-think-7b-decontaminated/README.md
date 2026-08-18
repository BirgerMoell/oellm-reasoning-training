# Dolci Think SFT 7B — decontaminated

| Field | Value |
|---|---|
| State | `production` |
| v1 allocation | 20.5% of rendered tokens |
| Role | complementary broad English reasoning traces |
| Public source | [`openeurollm/Dolci-Think-SFT-7B-decontaminated`](https://huggingface.co/datasets/openeurollm/Dolci-Think-SFT-7B-decontaminated) |
| Revision | `d6550429f7a59a28e8f0881ab51448a94fe771ca` |
| Rows | 2,267,351 |
| Format | Parquet; `messages`, `dataset_source`, `id` |
| License | composite; retain upstream source-level terms |
| LUMI raw location | `$OELLM_RUN_ROOT/raw/datasets/openeurollm--Dolci-Think-SFT-7B-decontaminated/` |

The OpenEuroLLM copy removes 827 benchmark-matching rows. It is selected after specialized sources and
the 32B Dolci pool during global prompt deduplication, which makes this large complementary pool absorb
the remaining duplicate loss instead of crowding out scarce domain or language data. Do not
flatten messages before applying the checkpoint’s Gemma-style template.
