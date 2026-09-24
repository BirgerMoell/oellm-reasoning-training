# Dolci Think SFT translated

| Field | Value |
|---|---|
| Public source | [`openeurollm/Dolci-Think-SFT-translated`](https://huggingface.co/datasets/openeurollm/Dolci-Think-SFT-translated) |
| Pinned revision | `ba4754ab30afb66e652c3690ef0390dcea4939cd` |
| License | Apache-2.0 |
| Upstream description | Gemma-4-31B-it machine translations of the successful/validated Dolci Think SFT 32B subset |
| Repository size | 1,008,455 rows, 11.8 GB |
| Languages | Czech, German, Greek, Spanish, Finnish, French, Italian, Dutch, Polish, Romanian, Swedish, Ukrainian |
| Schema | `{id, messages}` with one language/config and sharded Parquet files |
| Role | Explicit multilingual reasoning supervision in the Anneal-300B v2 continuation |

The v2 recipe treats every language as a separate token-weighted source with a 2% weighted-token quota.
This prevents high-resource language row counts from determining the mix and gives Swedish the same
budget as every other translated language. Rows must contain a complete, non-empty `<think>...</think>`
trace and a non-empty final answer, fit 64–16,384 rendered tokens without truncation, pass the strict
30-gram repetition rejection, and survive language-scoped normalized-prompt deduplication.

This is translated training data, not an independent correctness signal. The run therefore retains
verified OpenR1 math, independent Nemotron multilingual reasoning, the 37-language pilot, and 35%
Dolci Instruct replay. Evaluation must report language-specific accuracy and language fidelity rather
than treating translation volume as evidence of reasoning quality.
