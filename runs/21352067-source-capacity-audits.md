# Nemotron source-capacity audits — LUMI 2026-08-18

Result: **math and code require one-pass caps; STEM is sufficient**.

All audits used the production tokenizer, chat template, length bounds, adapters, source revisions,
and prompt hash. No audit artifact is training data, and no GPU was allocated.

| Job | Source / seed | Unique rows | Eligible tokens | Production quota | Result |
|---:|---|---:|---:|---:|---|
| `21352068` | math, no seed | 102,480 | 80,860,155 | 72,849,923 (3.5%) | sufficient |
| `21352067` | code, no seed | 33,846 | 56,065,276 | 41,628,528 (2%) | sufficient |
| `21351769` | code after prior build seed | 26,528 | 46,639,484 | 41,628,528 (2%) | sufficient, but less headroom |
| `21351837` | STEM after prior build seed | 333,859 | 173,142,274 | 145,699,846 (7%) | sufficient |

The audit tool exits nonzero when the configured quota exceeds capacity, so jobs `21352067` and
`21351769` originally ended with exit `2:0` while testing the old 8% code quota. Those exits are the
expected fail-closed result, not infrastructure failures. Job `21352068` completed `0:0`; its current
3.5% math quota passed. The earlier full-build attempt independently failed closed on the old 10% math
quota.

The final recipe selects the fixed pilot and scarce domain/language sources before broad Dolci. Math
and code remain below their no-seed one-pass capacities; Dolci 32B/7B then fill 27%/20.5% after absorbing
deduplication loss. This avoids repeated prompts while retaining the original top-level language mix.
