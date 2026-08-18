# OpenR1 Math 220K

| Field | Value |
|---|---|
| State | `production` |
| v1 allocation | 5% of rendered tokens |
| Role | independently verified mathematical solutions |
| Public source | [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) |
| Revision | `e4e141ec9dea9f8326f4d347be56105859b2bd68` |
| Configuration | `default`, 93,733 rows |
| License | Apache-2.0 |
| Shared LUMI file | `/scratch/project_462000963/datasets/posttraining_data/OpenR1-Math-220k/default-train.jsonl` |
| V1 snapshot | `$OELLM_RUN_ROOT/raw/datasets/open-r1--OpenR1-Math-220k/` |

The adapter keeps only rows with at least one positive verifier result, a completed reasoning trace,
and a complete `messages` conversation. If the pre-selected assistant message is invalid, the adapter
chooses the first generation marked correct and complete. It never trains all candidate generations.

This source is capped at 5% because the broader Dolci/Nemotron corpora already contain substantial math;
the purpose here is correctness pressure rather than domain dominance.
