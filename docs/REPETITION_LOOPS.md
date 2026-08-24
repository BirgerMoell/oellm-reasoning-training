# Repetition loops: diagnosis and mitigation plan

## Executive conclusion

The observed loops are not best explained by one broken setting. They are a failure chain:

1. at a difficult or uncertain reasoning point, the model assigns too little probability to a useful
   next step and relatively more to an easy cyclic action such as re-checking or restating;
2. greedy decoding always takes that locally most likely action;
3. the repeated text becomes part of the next context and increases the probability of repeating again;
4. the end-of-turn token loses every local argmax decision, so generation continues until the token cap.

This account is directly consistent with the published checkpoint examples and with controlled findings
on both general neural-text degeneration and reasoning-model loops. Sampling can help the decoder escape,
but it does not repair the learned probability error. A robust `reasoning-v2` therefore needs both safer
decoding and cleaner, shorter, more verifiable reasoning supervision.

## Evidence from this checkpoint

The five public examples were generated from the exact BF16 release with the native chat template,
`do_sample=False`, and a 768-new-token limit. The observed behavior is:

| Observation | Result | What it supports |
|---|---:|---|
| Completed by emitting `<end_of_turn>` | 3 / 5 | EOS and the chat template work in general |
| Hit the 768-token cap without EOS | 2 / 5 | once stalled, the model did not choose termination |
| Swedish literal phrase loop | 1 / 5 | exact textual self-reinforcement under greedy decoding |
| French semantic re-check loop | 1 / 5 | progress can stall even without one phrase repeating verbatim |
| Correct final answer | 1 / 5 | repetition is part of a broader reasoning-quality problem |
| Correct response language throughout | uneven | the reasoning continuation also shifted the learned language/style distribution |

The Swedish output correctly reaches `12/30 = 0.4 = 40%`, then repeats variants of the same check.
Its most frequent normalized 8-gram appears 15 times, its duplicate 4-gram ratio is 0.396, it never
closes `<think>`, and it does not emit EOS. The French output repeatedly rechecks a solved equation after
introducing an arithmetic error; it is primarily a **semantic** loop, so an exact n-gram detector alone
does not capture it.

The strict threshold used by Pipis et al.—any 30-gram occurring at least 20 times—does not flag either
768-token example: the Swedish maximum is 13 and the French maximum is 2. This is expected because the
published generations are capped much earlier than the long AIME traces studied in that paper. This
repository therefore reports that threshold using a documented Unicode lexical word/number normalization,
plus an explicitly labeled short-output warning (any 8-gram at least four times). Because the paper does
not specify the same normalization, the resulting rates are threshold-aligned rather than guaranteed
token-for-token reproductions. Neither metric substitutes for answer correctness or semantic review.

### Training-mixture evidence

The immutable LUMI manifest contains 1,130,994 conversations and 2,097,196,255 rendered tokens:

| Component | Manifest token share | Relevance to the hypothesis |
|---|---:|---|
| Dolci Think 32B + 7B | 47.14% | long synthetic teacher reasoning is the largest component |
| Nemotron reasoning slices | 32.26% | additional synthetic math/code/STEM/multilingual reasoning |
| OpenR1 verified math | 4.96% | answer-verified math reasoning |
| OpenEuroLLM multilingual pilot | 0.75% | multilingual teacher traces across 37 languages |
| Exact parent-SFT replay | 14.89% | ordinary instruction/capability retention |
| **All reasoning sources** | **85.11%** | reasoning style dominates the continuation |

There are 677,620 rows containing both `<think>` and `</think>` markers according to the build manifest.
Because rows vary greatly in length, this row count is not a token share. It also does not say whether a
trace is correct, concise, or repetitive. The reproducible sample audit below measures those properties.

This mixture makes transfer of teacher overthinking a credible contributor, not a proven sole cause.
The model is a 9B full-parameter SFT student learning traces from several teacher distributions, including
32B outputs. Research finds that distilled students can loop much more than their teachers even when the
training traces rarely contain literal loops: imperfectly learned progress actions are sufficient.

## Causal assessment

| Candidate cause | Confidence for this checkpoint | Evidence and caveat |
|---|---|---|
| greedy decoding triggers/amplifies the loop | **high** | both stalled traces used `do_sample=False`; this is the regime where reasoning-loop studies find the highest loop rate |
| repetition is contextually self-reinforcing | **high** | the Swedish phrase becomes increasingly entrenched, matching controlled probability measurements in prior work |
| the student learned cyclic “wait/re-check” actions more easily than progress actions | **medium-high** | trace shape matches the risk-aversion mechanism and synthetic teacher SFT dominates, but token probabilities were not logged for these examples |
| long-teacher-CoT SFT transferred overthinking | **medium** | the mixture is 85.11% reasoning tokens and the behavior matches long-CoT findings; a parent/model ablation is still required |
| repeated lexical spans in selected training rows contribute | **low-medium** | the audit found 7 strict candidates in 25,351 stratified rows, but several are repeated formula/translation patterns rather than demonstrated semantic reasoning loops |
| EOS/tokenizer is globally broken | **low** | EOS is `<end_of_turn>`, validation passed, and three of five examples stopped on it |
| prompt tokens were included in the loss | **low** | TRL assistant-only masking and the template generation mask were explicitly validated |
| corrupted BF16 weights | **very low** | all 9,101,947,904 values were finite and independent GPU generation passed |
| 256K RoPE or long-context extension caused these short-prompt loops | **low** | these prompts are short and this stage trained at 16K; no comparative evidence implicates RoPE |
| packed-example boundary leakage | **low** | the tested FlashAttention/packing path and masking gates passed; there is no boundary-specific symptom |

“Greedy caused it” and “training caused it” are not competing explanations. Training determines the
conditional token distribution; greedy decoding deterministically amplifies its local errors. Once the
first cycle enters the context, autoregressive feedback makes escape progressively less likely.

## What the research says

| Work | Finding relevant here | Operational implication |
|---|---|---|
| [Holtzman et al., *The Curious Case of Neural Text Degeneration* (ICLR 2020)](https://arxiv.org/abs/1904.09751) | likelihood-trained LMs combined with maximization decoding can produce unnaturally likely, repetitive text; nucleus sampling reduces degeneration | do not treat greedy as a neutral default for long free-form reasoning |
| [Welleck et al., *Neural Text Generation with Unlikelihood Training* (ICLR 2020)](https://arxiv.org/abs/1908.04319) | standard maximum-likelihood training overproduces frequent/repeated tokens; unlikelihood objectives reduce repetition under greedy and beam search | consider a targeted anti-repetition training loss after data fixes |
| [Chiang and Chen, *Relating Neural Text Degeneration to Exposure Bias* (BlackBoxNLP 2021)](https://arxiv.org/abs/2109.08705) | errors often precede degeneration; model-generated prefixes create exposure mismatch and self-reinforcing errors | evaluate long autoregressive rollouts, not only teacher-forced loss |
| [Xu et al., *Learning to Break the Loop* (NeurIPS 2022)](https://arxiv.org/abs/2206.02369) | the probability of repeating a sentence rises with previous repetitions; DITTO teaches models to penalize pseudo-repetitions | filter loops and test DITTO-style negative continuations |
| [Pipis et al., *Wait, Wait, Wait… Why Do Reasoning Models Loop?* (2025)](https://arxiv.org/abs/2512.12895) | low-temperature/greedy looping is common; smaller/distilled students loop more; hard progress actions and temporally correlated errors favor easy cyclic actions | sampling is a useful stopgap, but reduce student learning errors and cyclic supervision |
| [Yu et al., *Long-Short Chain-of-Thought Mixture SFT* (2025)](https://arxiv.org/abs/2505.03469) | direct long-CoT SFT inherits teacher overthinking; mixing structure-preserving short traces improved average accuracy and cut response length by about 47.6% in their experiments | build verified concise counterparts instead of training only on raw long traces |
| [Ghosal et al., *Does Thinking More Always Help?* (NeurIPS 2025)](https://arxiv.org/abs/2506.04210) | accuracy can rise and then fall with extended thinking; multiple independent shorter paths can beat one extended path at equal budget | compare parallel sampling/self-consistency with one very long trace |
| [Jung et al., *Code Execution as Grounded Supervision for LLM Reasoning* (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1260/) | verifiable execution-grounded traces reduced meaningless repetition and overthinking in their ablations | increase grounded/verifiable traces for math, code, and structured tasks |
| [Su and Collier, *Contrastive Search Is What You Need for Neural Text Generation* (2022)](https://arxiv.org/abs/2210.14140) | contrastive decoding can reduce degeneration without retraining | include it as an experimental decoder, subject to reasoning-accuracy checks |
| [Zhu et al., *Penalty Decoding* (EMNLP 2023)](https://arxiv.org/abs/2310.14971) | repetition penalties suppress self-reinforcement, but the penalty and history window require tuning | use only mild, measured penalties; avoid a blind hard ban |

The current Transformers documentation also notes that [greedy decoding can break down into repetition
on longer sequences](https://huggingface.co/docs/transformers/main/en/generation_strategies) and documents
the relevant [generation controls](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation).

## Reproducible audits

The audit reads the immutable selected Parquet, not a new download of an upstream dataset. It scans every
row to obtain source/language populations and analyzes the first 2,000 rows in every `(source, language)`
group. Because the materialized dataset was globally shuffled with a fixed seed and is identified by a
SHA-256, this is a deterministic stratified sample. It does not claim to be a full 2.1B-token lexical scan.

```bash
export OELLM_RUN_ROOT=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts
sbatch --export=ALL,OELLM_RUN_ROOT="$OELLM_RUN_ROOT" slurm/audit_repetition_lumi.sbatch
```

The output is written to:

```text
$OELLM_RUN_ROOT/audits/repetition/repetition-audit.json
```

The metrics are:

- `strict_loop_30gram_20x`: the Pipis et al. long-reasoning threshold, applied to the documented
  Unicode-normalized tokens;
- `short_repetition_8gram_4x`: a sensitive warning for shorter outputs, not a literature-equivalent loop rate;
- `repetition_4`: `1 - unique_4grams / total_4grams` after Unicode-aware case folding and removal of
  punctuation-only tokens;
- maximum 8-gram and 30-gram counts;
- complete, empty, unclosed, or absent `<think>` tags;
- the highest-repetition row identities and hashed repeated-span signatures per source, without copying
  training snippets into Git.

### LUMI audit result

Corrected lexical audit job `21493302` completed in 1m42s with exit code 0 and 288 MB maximum RSS. It
scanned all 1,130,994 rows for population accounting and analyzed 25,351 source/language-stratified rows.
The report is committed as [`runs/21493302-repetition-audit.json`](../runs/21493302-repetition-audit.json)
(SHA-256 `892d62d7ef3e024b6b9780470df7c4fa4c1ce96c4980445bcb617b987ff3e2db`). Training-span text is
not included; repeated-span signatures are hashed.

| Source | Sample | Strict 30-gram × 20 | Short 8-gram × 4 warning |
|---|---:|---:|---:|
| Dolci Think 32B | 2,000 | 0 (0%) | 721 (36.05%) |
| Dolci Think 7B | 2,000 | 1 (0.05%) | 587 (29.35%) |
| Exact SFT replay | 2,000 | 0 (0%) | 87 (4.35%) |
| Nemotron code | 2,000 | 0 (0%) | 74 (3.70%) |
| Nemotron German | 2,000 | 2 (0.10%) | 1,069 (53.45%) |
| Nemotron Spanish | 2,000 | 2 (0.10%) | 1,007 (50.35%) |
| Nemotron French | 2,000 | 1 (0.05%) | 1,015 (50.75%) |
| Nemotron Italian | 2,000 | 1 (0.05%) | 1,060 (53.00%) |
| Nemotron math | 2,000 | 0 (0%) | 180 (9.00%) |
| Nemotron STEM | 2,000 | 0 (0%) | 18 (0.90%) |
| OpenR1 verified math | 2,000 | 0 (0%) | 1,249 (62.45%) |
| Multilingual pilot | 3,351 (all) | 0 (0%) | 1,355 (40.44%) |
| **Stratified total** | **25,351** | **7 (0.0276%)** | **8,422 (33.22%)** |

Interpretation:

- The strict lexical candidates are rare but real patterns in the exact selected artifact. They occur in
  Dolci 7B and the four long Nemotron multilingual slices. Private inspection of short signatures shows
  formula and translation-pattern repetition among the strongest rows; the audit does **not** establish
  that all seven are semantic reasoning loops.
- The 0.0276% total is a source/language-stratified sample rate, not a population-weighted estimate for
  all 1.13M rows.
- The short warning fires on one third of the sample and rises with trace length and mathematical reuse.
  It is suitable for ranking manual review, not for automatically rejecting a third of the data.
- The sample contains 17,340 non-empty reasoning-tag rows, 5,603 empty `<think></think>` rows, and 2,408
  rows without think tags; it contains no unclosed tags. Empty tags are concentrated in Nemotron
  math/code/STEM and reveal a format mixture, not by themselves a loop defect.
- A punctuation-inclusive exploratory pass (`21493150`) initially marked 38 strict candidates, but its
  largest hits were Markdown/LaTeX separator runs. That result is retained on LUMI but rejected as the
  project metric. This is why the committed audit omits punctuation-only tokens and tests this behavior.

The data result weakens the simple theory that the model merely copied many literal prose loops. It does
not rule out learned overthinking or semantic cycling: the short French generation cycles through a
reasoning state without repeating one exact long phrase, and controlled research shows students can loop
far more often than their teacher traces.

## Mitigation plan

### 1. Establish a decoder matrix before more training

Run the parent SFT model and this checkpoint on the same independent multilingual reasoning set. For each
prompt, preserve raw output and score:

| Track | Initial sweep | Purpose |
|---|---|---|
| greedy control | `do_sample=false` | reproduce the deterministic failure regime |
| temperature | `temperature={0.2,0.4,0.6,0.8}`, `top_p=0.95`, at least 5 fixed seeds | measure escape, accuracy, and variance rather than assuming one setting |
| mild repetition penalty | `repetition_penalty={1.03,1.05,1.08}` crossed with the best sampling setting | test whether suppression helps without damaging equations/code |
| parallel reasoning | 4 shorter independent samples with verifier/majority selection at the same total token budget | compare useful exploration with one extended trace |
| contrastive search | one separately labeled experimental configuration | determine whether generic degeneration gains transfer to reasoning |

Do not make `no_repeat_ngram_size` a default. A hard ban can invalidate legitimate repeated symbols,
equations, citations, or code. A serving-side loop detector may stop an obvious cycle as a last-resort cost
control, but it should return a flagged incomplete response rather than silently presenting it as solved.

### 2. Repair the training data

For `reasoning-v2`:

1. reject strict exact loops and unclosed reasoning segments;
2. manually review or classifier-review the high short-repetition tail by source and language;
3. verify final answers and, where possible, intermediate states with execution or symbolic checkers;
4. create structure-preserving concise versions of long teacher traces and tune a long/short token mixture;
5. require the answer language to match the request unless translation is the task;
6. remove traces that repeatedly reach the same state without adding a premise, calculation, or subgoal;
7. retain enough exact SFT replay to recover instruction following, and measure that recovery rather than
   assuming 15% is sufficient.

Literal deduplication is necessary but not sufficient. The French example shows why the review also needs
semantic progress: a trace can cycle through re-checks while changing surface words.

### 3. Add an anti-loop training stage only after data repair

Compare small controlled continuations from the same repaired SFT checkpoint:

- ordinary assistant-only SFT on the repaired long/short mixture;
- DITTO-style pseudo-repetition negatives or unlikelihood loss;
- preference optimization on matched pairs where one continuation makes verified progress and the other
  restates/backtracks without progress;
- grounded supervision on executable math/code traces.

Do not infer that RL alone fixes looping. Pipis et al. find limited loop-rate change between the studied
Phi-4 reasoning/RL variants, and a stronger objective cannot compensate for unmeasured data defects.

### 4. Make repetition a release gate

Use at least 500 independent multilingual prompts, with difficult items oversampled. Report by language,
task, model, decoder, and seed:

- strict and short-warning exact-loop rates;
- semantic no-progress loop rate from blinded review or a validated classifier;
- EOS completion and max-token exhaustion rates;
- answer/verifier accuracy, instruction adherence, and correct-language rate;
- response length distribution and tokens per correct answer;
- first-loop position and whether an initially correct answer was lost through overthinking.

Provisional acceptance gates for the selected deployment decoder are: zero strict loops in 500 outputs,
less than 1% max-token exhaustion, no statistically clear answer-accuracy regression against the best
non-looping control, and recovery of the repository's existing instruction/language-retention gates. A
zero count in 500 samples still only puts the approximate one-sided 95% upper bound near 0.6%; it does not
prove that loops are impossible.

## Decision for this release

Keep `oellm-9b-256k-reasoning-v1` labeled **experimental**. The two published stalls are real model
failures, not merely presentation issues. Greedy decoding exposed them, but switching temperature does
not establish that the underlying reasoning distribution is repaired. The next decision should be based
on the controlled decoder matrix and the source-stratified training-data audit above.
