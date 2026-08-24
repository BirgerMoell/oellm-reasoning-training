# LUMI repetition audit 21493302

## Outcome

**Completed.** The corrected lexical audit scanned the complete selected reasoning-v1 Parquet and
analyzed a deterministic source/language-stratified sample. It found seven strict lexical repetition
candidates in 25,351 sampled rows (0.0276%). This is evidence of rare repeated spans, not a claim that all
seven are semantic reasoning loops or that the sample rate is population weighted.

## Execution

| Field | Value |
|---|---|
| Job | `21493302` |
| State | `COMPLETED`, exit `0:0` |
| Partition | LUMI `debug` |
| Elapsed | 00:01:42 |
| Allocated CPUs | 16 |
| Maximum RSS | 288,372 KiB |
| Input Parquet | `artifacts/data/reasoning-v1/train.parquet` |
| Input Parquet SHA-256 | `7cbb6ba4a69f457b50ffc89a124f20335719633c32e4ef47a3a844bbf48407ff` |
| Data manifest SHA-256 | `f92330319af5f917b6d9a01898d2a5dfc8f322b6d4250db1ce2111baafe077d1` |
| Audit script SHA-256 | `bf12abcad9d8427804bb19e14590e3146bc7d19444f1cafbd6dea090353a14d8` |
| Report SHA-256 | `892d62d7ef3e024b6b9780470df7c4fa4c1ce96c4980445bcb617b987ff3e2db` |

The private LUMI report is at:

```text
/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts/audits/repetition/repetition-audit-lexical.json
```

The same privacy-safe report is committed as
[`21493302-repetition-audit.json`](21493302-repetition-audit.json). It includes public model-example
phrases but replaces training-data repeated spans with SHA-256 signatures.

## Method

- scan all 1,130,994 rows to verify source/language population counts;
- take the first 2,000 rows per `(source_id, language)` from the deterministically shuffled, checksummed
  Parquet; the 3,351-row multilingual pilot is included in full because each language has fewer than
  2,000 rows;
- case-fold and tokenize Unicode lexical word/number spans, omitting punctuation-only tokens;
- strict candidate: a lexical 30-gram occurs at least 20 times;
- short warning: a lexical 8-gram occurs at least four times;
- report duplicate 4-gram ratio, tag state, and hashed signatures for the most repetitive training rows.

The resulting 25,351-row total is deliberately stratified. Aggregate percentages must not be interpreted
as population-weighted estimates.

## Source results

| Source | Rows | Strict candidates | Short warnings |
|---|---:|---:|---:|
| Dolci Think 32B | 2,000 | 0 | 721 |
| Dolci Think 7B | 2,000 | 1 | 587 |
| Exact SFT replay | 2,000 | 0 | 87 |
| Nemotron code | 2,000 | 0 | 74 |
| Nemotron German | 2,000 | 2 | 1,069 |
| Nemotron Spanish | 2,000 | 2 | 1,007 |
| Nemotron French | 2,000 | 1 | 1,015 |
| Nemotron Italian | 2,000 | 1 | 1,060 |
| Nemotron math | 2,000 | 0 | 180 |
| Nemotron STEM | 2,000 | 0 | 18 |
| OpenR1 verified math | 2,000 | 0 | 1,249 |
| Multilingual pilot | 3,351 | 0 | 1,355 |
| **Total** | **25,351** | **7** | **8,422** |

Tag states were 17,340 non-empty complete think segments, 5,603 empty think segments, and 2,408 rows
without tags. No sampled row had an unclosed think segment.

## Measurement correction

Exploratory job `21493150` used punctuation-inclusive n-grams and reported 38 strict candidates. Its
largest signatures were repeated Markdown table bars and separator dashes, demonstrating a false-positive
mode for technical reasoning data. That raw report remains on LUMI with SHA-256
`fc8c7e112df42a76f59da60e222b25eddf6184a558ffc5c2d0934d46b9bc0d82`, but it is not the accepted metric.

Job `21493302` removed punctuation-only tokens. Review of the corrected candidate signatures still finds
formula and translation-pattern repetition, so the seven rows require semantic review before being called
reasoning loops. The short 8-gram warning is even more length-sensitive and is only a triage ranker.

## Decision impact

The audit does not support “the student copied a large number of literal prose loops” as the primary
explanation. It does support filtering/reviewing a small high-repetition tail and normalizing inconsistent
think-tag formats in reasoning-v2. The stronger checkpoint-specific explanation remains: learned
overthinking/progress errors are amplified by greedy decoding and then self-reinforced by the generated
prefix. See [`../docs/REPETITION_LOOPS.md`](../docs/REPETITION_LOOPS.md) for the cited causal analysis and
mitigation experiment.
