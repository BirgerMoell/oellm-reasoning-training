# Evaluation and acceptance gates

The same committed harness revision, prompts, few-shot examples, chat template, decoding parameters, and
judge model must score the baseline and candidate. Store raw generations and task-level results.

## Matrix

| Capability | Benchmarks | Primary measure |
|---|---|---|
| math | GSM8K, MATH-500, AIME 2024/2025/2026, AMC23 | exact/verifier accuracy |
| scientific reasoning | GPQA Diamond, JEEBench | accuracy |
| code | HumanEval+, MBPP+, LiveCodeBench current frozen release | pass@1 |
| instruction following | IFEval, project EU holdouts, AlpacaEval-style rubric | strict accuracy / win rate |
| multilingual reasoning | MGSM in available European languages; translated, reviewed math set | exact accuracy by language |
| language retention | fixed prompts in all 37 model-card languages | correct-language rate and quality rubric |
| safety | the same safety/refusal set used for the SFT baseline | safe/helpful classification |
| long context | natural-word NIAH at 8K/32K/128K/256K; RULER 4K–64K | exact retrieval / aggregate |
| format | 500 mixed prompts | valid turn termination, no template tokens in visible text |

Do not add benchmark training examples to the mixture after baseline evaluation. The pinned OpenEuroLLM
datasets already remove matches against the core suite; keep their decontamination metadata in the run
record.

## Relative gates

Let `B` be the score from the pinned starting checkpoint and `C` the candidate scored with the same run.
The checkpoint is accepted only if all required gates pass:

| Gate | Requirement |
|---|---|
| MATH-500 | `C >= B + 8` percentage points |
| GSM8K | `C >= B + 5` points |
| reasoning aggregate | geometric mean improves by at least 10% relative |
| GPQA Diamond | no more than 2 points below baseline |
| code aggregate | no more than 2 points below baseline; at least one code task improves |
| EU holdouts overall | no more than 2 points below baseline |
| instruction following | no more than 3 points below baseline |
| European-language reasoning | mean improves by at least 5 points; no trained language drops more than 2 |
| 37-language retention | correct-language response rate at least 98%; no high-resource language drops more than 3 rubric points |
| safety | no statistically clear regression on the fixed set |
| natural-word NIAH | retain 40/40 across the published grid |
| architecture | exact match for positions, RoPE theta, vocab size, special token IDs, and chat markers |

A checkpoint that gives a large reasoning gain but misses a retention gate remains a named experimental
artifact. It can motivate `reasoning-v2`; it does not replace the production checkpoint.

## Checkpoint selection

Score steps 500, 1,000, 1,500, and 2,000. First apply hard retention gates. Among survivors, rank by the
geometric mean of normalized math, scientific reasoning, code, English reasoning, and European-language
reasoning. Report every checkpoint so selection is auditable.

## Generation settings

Maintain two tracks:

- deterministic: temperature 0, one completion, strict answer extraction;
- reasoning sampling: the committed task-recommended temperature/top-p and fixed number of samples.

Set generation EOS to `<end_of_turn>`. Cap outputs by task rather than using one short universal cap;
reasoning can appear worse solely because the final answer was truncated.

## Required outputs

For every model/task pair keep:

- model path/revision and model-config SHA;
- harness Git SHA and environment versions;
- task revision/split and decontamination status;
- exact generation configuration and seed;
- raw prompt/response/score rows where licensing permits;
- aggregate JSON and stderr/bootstrap confidence interval;
- elapsed GPU time and failures/timeouts.
