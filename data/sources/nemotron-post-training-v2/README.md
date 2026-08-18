# Nemotron Post-Training Dataset v2

| Field | Value |
|---|---|
| State | `production` for decontaminated snapshot; `available` for shared raw copy |
| v1 allocation | 38.5% total: math 3.5%, code 8%, STEM 7%, `de/fr/es/it` 5% each |
| Public source | [`openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated`](https://huggingface.co/datasets/openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated) |
| Revision | `789a044d6f305996098ef340b4264cfc022ed12a` |
| License | CC-BY-4.0 |
| Format | Parquet; `messages` plus UUID, generator, category, reasoning metadata |
| LUMI raw source | `/scratch/project_462000963/datasets/posttraining_data/Nemotron-Post-Training-Dataset-v2/` (31 GiB; not decontaminated) |
| LUMI v1 snapshot | `$OELLM_RUN_ROOT/raw/datasets/openeurollm--Nemotron-Post-Training-Dataset-v2-decontaminated/` |

The OpenEuroLLM snapshot has distinct `math`, `code`, `stem`, `chat`, and multilingual splits. V1 does
not train the general `chat` split because the replay allocation already covers instruction behavior.
It also does not use `ja` because the explicit multilingual allocation is reserved for four European
languages. Each split has an independent token quota so high-volume multilingual splits cannot drown
out math or code.

Full-build attempt `21350927` measured 80,930,889 eligible math tokens after filtering and cross-source
deduplication, below the original 10% quota of 208,142,638. V1 therefore caps the math slice at 3.5%
(about 72.85M tokens) and reallocates the difference to the two decontaminated Dolci pools. This preserves
the 65% English-reasoning allocation without repeating math prompts merely to fill a nominal share.

The shared LUMI raw files remain useful for audits and historical reproduction:

- `math.jsonl`: 26.55 GB
- `code.jsonl`: 336.5 MB
- `science.jsonl`: 6.02 GB
- `chat.jsonl`: 255.1 MB
- `safety.jsonl`: 57.7 MB

Do not substitute those raw files silently; their row set and schema differ from the pinned production
snapshot and would invalidate the run manifest.
