# LUMI run records

This directory records completed execution gates and production candidates. Large logs,
checkpoints, and datasets remain under the LUMI artifact root; each record contains the exact
locations and provenance needed to inspect them.

| Date | Job | Purpose | Result |
|---|---:|---|---|
| 2026-08-24 | [`21492473`](oellm-9b-256k-reasoning-v1-release.md) | BF16 release validation and Hugging Face publication | passed; public experimental model published |
| 2026-08-19 | [`21366870`](21366870-reasoning-v1.md) | full reasoning-v1 production training | completed; step 2,000 selected for experimental release |
| 2026-08-18 | [`21352067` group](21352067-source-capacity-audits.md) | math/code/STEM one-pass capacity audits | passed; capacity-safe quotas established |
| 2026-08-18 | [`21350927`](21350927-reasoning-data-attempt.md) | first full reasoning-v1 data build | failed safely; measured math capacity and corrected recipe |
| 2026-08-18 | [`21348717`](21348717-reasoning-sanity.md) | eight-node production-path reasoning sanity | passed |
