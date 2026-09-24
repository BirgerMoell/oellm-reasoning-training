# MedQA reasoning traces

| Field | Value |
|---|---|
| Public source | [`birgermoell/medqa-reasoning-traces`](https://huggingface.co/datasets/birgermoell/medqa-reasoning-traces) |
| Pinned revision | `82fe04a7165fdafc5bacb7f6a988d4d0763d6d9a` |
| Upstream questions | MedQA `med_qa_en_4options_source`, English USMLE-style multiple choice |
| Trace generator | `kimi-k3`, temperature 0 |
| Training rows available | 10,178; 9,598 marked correct before completion/length gates |
| License | `medqa-research-use`; downstream use must preserve and review the source restrictions |
| Role | Small verifier-filtered medical reasoning specialization slice |

Only the original `train` split is eligible. Validation and test questions are deliberately excluded to
preserve MedQA evaluation integrity. The adapter selects rows whose parsed answer equals the gold label,
requires `finish_reason == stop`, formats the question and A–D options as the user prompt, and constructs
`<think>{reasoning}</think>{response}` as the assistant completion. The standard 64–16,384-token,
strict-repetition, full-answer, language-scoped deduplication, and assistant-only-loss gates then apply.

This is model-generated reasoning over research-use medical exam material, not clinically adjudicated
explanation data and not suitable for clinical advice. A correct option letter does not guarantee every
intermediate medical claim is correct. The v2 recipe limits it to 2% of weighted tokens and requires
domain-specific evaluation before any release claim.
