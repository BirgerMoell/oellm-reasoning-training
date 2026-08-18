# Reasoning-v1 data build attempt — LUMI job 21350927

Result: **failed safely at a source-capacity gate**. No training job was submitted and no partial
artifact is valid for training.

## Execution

| Field | Value |
|---|---|
| Date | 2026-08-18 |
| Repository at submission | `94f67abd39022481ca73797e2e308b9483bc155e` |
| Slurm state | `FAILED`, exit `1:0` |
| Elapsed | 1h02m49s |
| Allocation | one `small` node, 96 requested CPUs, 480 GiB memory |
| Peak RSS | 259,525,816 KiB |
| Partial artifact archive | `artifacts/data/reasoning-v1.failed-21350927/` |

## Proven observations

- multilingual pilot: 3,351 rows and 15,725,624 tokens, exactly matching its fixed invariants;
- Dolci 32B: 87,912 selected rows and 416,286,344 tokens for a 416,285,275-token quota;
- Dolci 7B: 73,755 selected rows and 312,216,154 tokens for a 312,213,956-token quota;
- Nemotron math: 80,930,889 eligible tokens after filtering and cross-source deduplication.

The original 10% Nemotron-math quota was 208,142,638 tokens, so the builder stopped rather than
silently oversampling duplicate math prompts or producing a short mixture.

## Recipe correction

Production v1 caps Nemotron math at 3.5%, safely below its measured capacity. Follow-up audits also
found only 56,065,276 one-pass unique code tokens, so code is capped at 2%. Specialized and language-
targeted sources now claim duplicate prompts before the broad Dolci pools, which are increased to 27%
and 20.5% to absorb both capacity shortfalls. The overall allocation remains 65% English reasoning,
20% `de/fr/es/it` reasoning, and 15% exact SFT replay. A new full build must pass all remaining source
quotas, the manifest checksum checks, and the full prompt-hash deduplication scan.
