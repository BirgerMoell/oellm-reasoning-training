# OpenEuroLLM multilingual reasoning traces — v0.2 pilot

| Field | Value |
|---|---|
| State | `production-pilot`, consume every valid 16K-eligible accepted row once |
| v1 allocation | fixed coverage floor, not a percentage and never oversampled |
| Role | mathematical reasoning coverage across 37 non-English target languages |
| Public source | [`openeurollm/reasoning-traces-multilingual`](https://huggingface.co/datasets/openeurollm/reasoning-traces-multilingual) |
| Revision | `031163d3a8876682a395cc65abf924b1edc2fcd9` |
| Release | `v0.2-pilot` |
| Rows | 3,425 accepted translations from 99 source problems; 3,351 fit the v1 16K limit |
| Selected rendered tokens | 15,725,624 with the pinned checkpoint tokenizer (0.750% of v1) |
| Languages | `bg, bs, ca, cs, cy, da, de, el, es, et, eu, fi, fr, ga, gl, hr, hu, is, it, lb, lt, lv, mk, mt, nl, no, pl, pt, ro, ru, sk, sl, sq, sr, sv, tr, uk` |
| Format | Parquet; structured `messages`, language, upstream provenance, translation metrics, quality gates |
| License | CC-BY-4.0; retain upstream attribution |
| LUMI v1 snapshot | `$OELLM_RUN_ROOT/raw/datasets/openeurollm--reasoning-traces-multilingual/` |

The dataset translates a deterministic 100-problem Llama-Nemotron math sample into 37 languages. One
source problem produced no accepted translation because of malformed `<think>` tags. Published rows have
already passed automated language, script, protected-token, structural, length, and source-copy gates.

The recipe additionally requires `quality.accepted == true`, complete user/assistant messages, a usable
reasoning trace, and a complete rendered length of 64–16,384 tokens. The pinned checkpoint tokenizer renders
74 accepted rows above that limit (maximum 27,103), so v1 rejects them rather than truncating a reasoning
trace or final answer. It preserves the row’s real language field and deduplicates within language. Every
surviving accepted row must be selected; the build fails unless it sees 3,351 selected rows, 15,725,624
rendered tokens, 74 `too_long` exclusions, and 37 selected languages.

This is deliberately a coverage floor rather than a token share. Repeating 99 underlying math problems to
fill a large percentage would teach prompt repetition rather than broad multilingual reasoning. The larger
Dolci/Nemotron/OpenR1 slices provide reasoning volume.

The upstream card states that this remains a pilot without systematic native-speaker review or independent
downstream-solver verification. Keep those limitations in the run record and evaluate reasoning-language
consistency by language before promoting the trained checkpoint.
