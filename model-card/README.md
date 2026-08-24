---
license: apache-2.0
base_model: birgermoell/oellm-9b-256k-sft
library_name: transformers
pipeline_tag: text-generation
tags:
- openeurollm
- qwen3
- reasoning
- multilingual
- long-context
- 256k
- supervised-fine-tuning
- experimental
datasets:
- openeurollm/reasoning-traces-multilingual
- openeurollm/Dolci-Think-SFT-32B-decontaminated
- openeurollm/Dolci-Think-SFT-7B-decontaminated
- openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated
- open-r1/OpenR1-Math-220k
language: [en, sv, de, fr, es, it, nl, pl, pt, cs, fi, da, el, bg, hr, hu, ro, sk, sl, et, lt, lv, ga, mt, eu, gl, is, nb, nn, sr, uk, ca, mk, sq, oc, lb, bs, cy, tr, ru]
---

# OELLM 9B 256K Reasoning v1

This is the **experimental step-2,000 checkpoint** from a full-parameter reasoning-SFT
continuation of [`birgermoell/oellm-9b-256k-sft`](https://huggingface.co/birgermoell/oellm-9b-256k-sft).
It was trained on LUMI using 2.086 billion packed input tokens from a deterministic multilingual
reasoning mixture.

This release is published for analysis and follow-on post-training. It is **not an accepted
production upgrade** over the parent SFT model: the measured checkpoint improves ARC-Challenge
and flexible-answer multilingual MGSM, but regresses English GSM8K, IFEval, and MMLU college
computer science. It has not passed the training repository's retention gates.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "birgermoell/oellm-9b-256k-reasoning-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [{
    "role": "user",
    "content": "En låda innehåller 18 röda och 12 blå kulor. Vad är sannolikheten att dra en blå kula? Resonera steg för steg.",
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The configuration retains `max_position_embeddings=262144` and RoPE theta 64,000,000. This stage
trained at 16,384 tokens and did not re-evaluate long-context retrieval. Treat 262K as an
architectural limit, not a demonstrated reasoning length for this checkpoint. Inference near the
limit requires substantial multi-GPU KV-cache memory.

## Verbatim reasoning examples

These are deterministic outputs from this exact published BF16 checkpoint, generated on an AMD
MI250X by LUMI job `21492664`. Each prompt was run independently with the native chat template, no
system prompt, greedy decoding (`do_sample=False`), seed `20260824`, and a limit of 768 new tokens.
The prompts were fixed before generation. The assessment column was added by manually checking the
output against the reference answer; outputs were not edited to make the model look better.

| Language / task | Reference answer | Observed result | Assessment |
|---|---|---|---|
| English, purchase/change | €4.25 | €4.25; stopped after 121 tokens | **Correct** |
| Swedish, probability | `2/5 = 40 %` | derived `12/30 = 0.4 = 40%`, then repeated itself until the 768-token limit without a final answer | **Incomplete** |
| German, rate/distance | 40 km | 8,100 km | **Incorrect**; misread 90 km in 1.5 hours as 90 km/h and mixed hours with minutes |
| French, linear equation | `x = 7` | derived `x = 7`, then made an arithmetic error while checking it and looped until the limit | **Incomplete / unstable** |
| Spanish, heads/legs | 6 rabbits and 8 chickens | 3 rabbits and 8 chickens; final box contained only `3` | **Incorrect** |

The complete unabridged artifact, including all five prompts, raw outputs, parsed traces/answers,
token counts, stop conditions, and per-output SHA-256 hashes, is in
[`examples.json`](examples.json) (file SHA-256
`890cac7bcf2048f0db852ba8a356a2ec4809a959d5e264365d5523a1c99f4c37`). The three complete
generations are reproduced below, together with the truncated Swedish trace requested for inspection.
The French sample is retained in full in the JSON but is not duplicated here because it ends in a
long repetition loop.

<details>
<summary><strong>English — correct, direct step-by-step output</strong></summary>

**Prompt**

> A shop sells notebooks for €4.50 each. Lina buys 3 notebooks and also pays €2.25 for a pen. She
> pays with €20. How much change should she receive? Reason step by step, then clearly state the
> final answer.

**Verbatim model output**

```text
Lina buys 3 notebooks for €4.50 each, so she pays €4.50 * 3 = €13.50.
She also pays €2.25 for a pen.
So she pays a total of €13.50 + €2.25 = €15.75.
She pays with €20, so she receives €20 - €15.75 = €4.25 change.
Thus, Lina should receive \boxed{4.25} euros in change.
```

</details>

<details>
<summary><strong>Swedish — truncated explicit reasoning trace, no final answer</strong></summary>

**Prompt**

> En låda innehåller 18 röda och 12 blå kulor. En kula dras slumpmässigt. Vad är sannolikheten att kulan är blå? Svara både som ett förkortat bråk och i procent. Resonera steg för steg och ange sedan slutsvaret tydligt.

**Verbatim `<think>` reasoning trace**

The generation reached the 768-token limit before emitting `</think>` or a separate answer.

```text
Okay, let's see. The problem is about finding the probability that a randomly drawn ball is blue from a box with 18 red and 12 blue balls. Hmm, probability questions can be tricky, but let me break it down step by step.

First, I need to figure out the total number of balls in the box. The question says there are 18 red and 12 blue. So, adding those together gives 18 + 12 = 30 balls in total. That's the denominator in the probability fraction.

Now, the numerator is the number of blue balls. The problem states there are 12 blue balls. So, the probability should be 12 divided by 30. Let me write that as a fraction: 12/30.

Wait, but the question asks for the probability both as a fraction and in percentage. So, first, I need to convert the fraction to a percentage. To do that, I can multiply the fraction by 100. So, 12/30 times 100. Let me calculate that. 12 times 100 is 1200, and 30 times 100 is 3000. So, 1200/3000 simplifies to 60/150, which is 3/7.5. Wait, that can't be right. Wait, 1200 divided by 3000 is 60/150, which is 3/7.5. But 3/7.5 is the same as 6/15, which is 2/5. Hmm, maybe I made a mistake here. Let me check again.

Wait, 12/30 is 0.4, which is 40%. So, 12/30 is 40%? Wait, no. Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40%? Wait, 12 divided by 30 is 0.4, which is 40
```

The trace reaches the correct value `12/30 = 0.4 = 40%`, but then enters a repetition loop. The
reference answer is `2/5 = 40%`; the model did not produce a completed final-answer segment. The
visible reasoning also switches to English even though the user prompt is Swedish.

</details>

<details>
<summary><strong>German — complete explicit reasoning trace, incorrect answer</strong></summary>

**Prompt**

> Ein Zug fährt mit gleichbleibender Geschwindigkeit 90 km in 1,5 Stunden. Wie weit fährt er in
> 40 Minuten? Begründe deine Rechnung Schritt für Schritt und nenne dann das Endergebnis klar.

**Verbatim `<think>` reasoning trace**

```text
Okay, let's see. The problem is about a train traveling at a constant speed of 90 km/h for 1.5 hours. I need to find out how far it goes in 40 minutes. Hmm, first, I should convert the time from hours to minutes because the speed is given in km/h, and the time is in hours. Right, 1 hour is 60 minutes, so 1.5 hours is 1.5 multiplied by 60. Let me calculate that. 1.5 times 60 is 90 minutes. So the time is 90 minutes.

Now, the distance is speed multiplied by time. So 90 km/h times 90 minutes. Let me do that multiplication. 90 times 90. Let's see, 90 times 100 is 9000, but since it's 90 times 90, that's 8100. Wait, 90 times 90 is 8100. So the distance is 8100 km. Let me check that again. 90 times 90: 90*90. 90*100 is 9000, minus 90*10 is 900, so 9000 - 900 is 8100. Yeah, that's right.

So the answer should be 8100 km. Let me make sure I didn't make any mistakes in the conversion or the calculation. The time was converted correctly from 1.5 hours to 90 minutes. Then the speed is 90 km/h, multiplied by 90 minutes. The calculation seems straightforward. I think that's right.
```

**Verbatim answer after `</think>`**

```text
Der Zug fährt mit einer konstanten Geschwindigkeit von 90 km/h für 1,5 Stunden. Um die Zeit in Minuten umzurechnen, multipliziere man 1,5 Stunden mit 60 Minuten/Stunde: 1,5 × 60 = 90 Minuten. Die Distanz ist die Geschwindigkeit (90 km/h) mal die Zeit (90 Minuten). Berechne 90 × 90: 90 × 100 = 9000, minus 90 × 10 = 900, also 8100 km.

**Antwort:** Der Zug fährt **8100 km** in 40 Minuten.
```

The correct calculation is `(90 km / 90 min) × 40 min = 40 km`.

</details>

<details>
<summary><strong>Spanish — structured reasoning without think tags, incorrect answer</strong></summary>

**Prompt**

> En una granja hay gallinas y conejos. En total hay 14 cabezas y 40 patas. ¿Cuántos conejos y
> cuántas gallinas hay? Razona paso a paso y termina con una respuesta final clara.

**Verbatim model output**

```text
## Paso 1: Determinar el número de patas de las gallinas y los conejos.
Hay 14 cabezas en total, y cada gallina tiene 2 patas y cada conejo tiene 4 patas.

## Paso 2: Calcular el número total de patas.
Primero, calcule el número total de patas de las gallinas: 14 gallinas * 2 patas/gallina = 28 patas.
Luego, calcule el número total de patas de los conejos: 40 patas - 28 patas (de las gallinas) = 12 patas.

## Paso 3: Calcular el número de conejos.
Cada conejo tiene 4 patas, por lo que el número total de conejos es 12 patas / 4 patas/conejo = 3 conejos.

## Paso 4: Calcular el número de gallinas.
El número total de patas es 28 patas, y el número total de patas de las gallinas es 28 patas - 12 patas (de los conejos) = 16 patas.
Cada gallina tiene 2 patas, por lo que el número total de gallinas es 16 patas / 2 patas/gallina = 8 gallinas.

## Paso 5: Determinar el número total de conejos y gallinas.
El número total de conejos es 3 y el número total de gallinas es 8.

La respuesta final es: $\boxed{3}$
```

The correct simultaneous-equation solution is 6 rabbits and 8 chickens. The model's result also
fails its own 14-head constraint.

</details>

## Model lineage and architecture

| Field | Value |
|---|---|
| Released checkpoint | reasoning-v1 step 2,000 |
| Parent | `birgermoell/oellm-9b-256k-sft@08359ad61333263c067edaf290067fea5b103d34` |
| Earlier base | `openeurollm/oellm-9b-256k-theta64m-prelude` |
| Architecture | dense `Qwen3ForCausalLM` |
| Parameters | 9,101,947,904 |
| Layers / hidden size | 36 / 4,096 |
| Attention heads / KV heads | 32 / 8 |
| Vocabulary | 263,168 |
| Context configuration | 262,144 tokens; RoPE theta 64M |
| Weight format | unquantized BF16 safetensors |
| Turn format | Gemma-style `<start_of_turn>` / `<end_of_turn>` template |

## Training data

The immutable builder selected 1,130,994 unique conversations and 2,097,196,255 rendered tokens.
Complete conversations outside 64–16,384 tokens were rejected rather than truncating an answer.
Selection was token-budgeted, globally shuffled with seed `20260818`, and deduplicated using a
language-scoped SHA-256 of the normalized user prompt. Specialized and language-targeted sources
claimed duplicate prompts before the broad Dolci pools.

| Materialized slice | Rows | Rendered tokens | Share | Terms recorded in manifest |
|---|---:|---:|---:|---|
| OpenEuroLLM multilingual reasoning traces v0.2 pilot (37 non-English languages) | 3,351 | 15,725,624 | 0.75% | CC-BY-4.0 |
| Nemotron v2 math, decontaminated | 91,910 | 72,850,226 | 3.47% | CC-BY-4.0 |
| Nemotron v2 code, decontaminated | 26,020 | 41,630,061 | 1.99% | CC-BY-4.0 |
| Nemotron v2 STEM, decontaminated | 281,012 | 145,700,137 | 6.95% | CC-BY-4.0 |
| Nemotron v2 German | 17,730 | 104,077,800 | 4.96% | CC-BY-4.0 |
| Nemotron v2 French | 17,896 | 104,077,810 | 4.96% | CC-BY-4.0 |
| Nemotron v2 Spanish | 18,310 | 104,085,146 | 4.96% | CC-BY-4.0 |
| Nemotron v2 Italian | 17,734 | 104,071,448 | 4.96% | CC-BY-4.0 |
| OpenR1 Math 220K, verified solutions | 17,343 | 104,071,331 | 4.96% | Apache-2.0 |
| Dolci Think 32B, decontaminated | 125,534 | 561,988,057 | 26.80% | ODC-By-1.0 |
| Dolci Think 7B, decontaminated | 116,995 | 426,703,655 | 20.35% | composite; see upstream |
| Exact prior-SFT mixture replay | 397,159 | 312,214,960 | 14.89% | composite; see parent card |

Pinned revisions and per-source processing are in the
[`reasoning-v1` recipe](https://github.com/BirgerMoell/oellm-reasoning-training/blob/main/configs/data/reasoning-v1.yaml)
and its [human-readable source cards](https://github.com/BirgerMoell/oellm-reasoning-training/tree/main/data/sources).
The materialized manifest SHA-256 is
`f92330319af5f917b6d9a01898d2a5dfc8f322b6d4250db1ce2111baafe077d1`; the 6.83 GB training
Parquet SHA-256 is `7cbb6ba4a69f457b50ffc89a124f20335719633c32e4ef47a3a844bbf48407ff`.

Dataset terms are not replaced by this model repository's Apache-2.0 weight license. The replay
slice inherits a composite lineage from Tulu 3 and EuroBlocks, and users should review all linked
upstream cards for their use case.

## Training procedure

This is assistant-only supervised fine-tuning, not RL, RLVR, DPO, or GRPO. TRL received structured
conversations and a template with assistant-generation masks; prompt/user tokens were excluded from
the loss. Conversations were packed after rendering.

| Hyperparameter | Value |
|---|---|
| Method | full-parameter TRL SFT, assistant-only loss |
| Sequence length | 16,384 |
| Global sequence batch | 64 (one sequence per GCD, no gradient accumulation) |
| Optimizer steps | 2,000 (0.9945 effective epoch) |
| Packed input tokens reported | 2.086 billion |
| Optimizer | AdamW, zero weight decay |
| Learning-rate schedule | peak `3e-6`, cosine decay, 3% warmup |
| Precision | BF16 training; BF16 release export |
| Memory/attention | FSDP, gradient checkpointing, FlashAttention 2, fused linear cross-entropy |
| Hardware | 8 LUMI-G nodes, 64 AMD MI250X GCDs |
| Slurm job | `21366870` (`COMPLETED`, exit `0:0`) |
| Runtime / allocation | 10:36:12 / 678.61 GCD-hours |
| Mean training loss | 0.8245 |
| First → final logged loss | 0.9382 → 0.8172 |
| First → final mean token accuracy | 0.7418 → 0.7646 |

The job ran from 2026-08-19 16:58 EEST to 2026-08-20 03:34 EEST. Framework versions were
TRL 0.28.0, Transformers 5.12.1, PyTorch 2.9.1+ROCm 6.4, Datasets 5.0.0, and Tokenizers 0.22.2.
The exact training repository commit was `f5621b4d7dd211499693dea918339ac92ccfd438`.

## Evaluation

### Protocol

The parent and checkpoints at steps 500, 1,000, 1,500, and 2,000 were evaluated with
lm-evaluation-harness 0.4.11, Transformers 5.2.0, PyTorch 2.7.1+ROCm, BF16, batch size 1,
the native chat template, greedy decoding, and seed `20260821`. All runs saved raw generations.
The parent and candidate used identical prompts and decoding. Full array: LUMI job `21443216`.

These results are useful for checkpoint comparison, not clean estimates of generalization. The
parent's earlier SFT mixture contains math/instruction datasets related to some benchmarks; the
reasoning-v1 additions use decontaminated OpenEuroLLM copies, but the 15% exact replay preserves the
parent data lineage.

### Parent vs released step 2,000

Scores are percentages. Arrows show whether higher is better (all listed metrics: higher is better).

| Benchmark / metric | Parent SFT | Step 2,000 | Δ |
|---|---:|---:|---:|
| GSM8K 4-shot, strict exact match | 30.78 | 29.34 | -1.44 |
| GSM8K 4-shot, flexible extract | 30.93 | 29.57 | -1.36 |
| ARC-Challenge 25-shot, normalized accuracy | 50.09 | 51.88 | +1.79 |
| ARC-Challenge 25-shot, accuracy | 44.97 | 46.76 | +1.79 |
| IFEval prompt-level strict | 41.04 | 26.62 | -14.42 |
| IFEval instruction-level strict | 52.64 | 39.21 | -13.43 |
| MMLU college computer science 5-shot, accuracy | 51.00 | 48.00 | -3.00 |
| MGSM German 0-shot, flexible extract | 20.80 | 29.60 | +8.80 |
| MGSM Spanish 0-shot, flexible extract | 24.00 | 27.60 | +3.60 |
| MGSM French 0-shot, flexible extract | 20.40 | 24.40 | +4.00 |
| MATH-500 0-shot, exact match | 0.40 | pending extended rerun | — |

MGSM strict-match scores were 0.0–0.4% for both parent and candidates because responses rarely
matched the benchmark's strict output form; flexible extraction is reported above, and both metrics
remain in the raw evaluation artifacts.

### Checkpoint sweep (seven completed tasks)

The partial macro below is the unweighted mean of configured primary metrics over GSM8K,
ARC-Challenge, IFEval, MMLU college computer science, and German/Spanish/French MGSM. MATH-500 is
excluded because its four candidate jobs exceeded the initial 12-hour limit. This is a diagnostic
summary, not a release gate.

| Checkpoint | Seven-task partial macro |
|---|---:|
| Parent SFT | 29.83 |
| Step 500 | 28.44 |
| Step 1,000 | 27.97 |
| Step 1,500 | 28.75 |
| **Step 2,000 (this release)** | **28.67** |

The published step is the final training checkpoint requested for release; it is not presented as
the best aggregate checkpoint. The extended candidate MATH-500 rerun is tracked as LUMI job
`21483191`, and the card will be updated if those results complete.

### Not evaluated

- GPQA Diamond was not run because the evaluation account did not have access to the gated dataset.
- MBPP was not claimed because the pinned harness container's code-execution metric did not
  initialize reliably. MMLU college computer science is only a code-related knowledge check, not a
  substitute for executable code generation.
- Long-context retrieval, safety/red-teaming, factuality, calibration, and broad multilingual
  instruction following were not re-evaluated on this checkpoint.

## Validation and reproducibility

- Step-2,000 audit job `21426582` exhaustively scanned 399 tensors and 9,101,947,904 values:
  `nonfinite=0`.
- The released BF16 export is independently checked for uniform BF16 dtype, architecture/tokenizer
  invariants, all-value finiteness, finite GPU logits, and non-empty English/Swedish/German generation.
- Export manifest and validation reports are included as `export_manifest.json` and
  `validation.json` in this repository.
- Training/evaluation code and the detailed run record are available in
  [`BirgerMoell/oellm-reasoning-training`](https://github.com/BirgerMoell/oellm-reasoning-training).

## Intended use

- Research on multilingual reasoning-SFT mixtures and checkpoint behavior.
- A reproducible starting point for controlled preference optimization, RLVR, verifier training,
  or capability-repair experiments.
- Comparative evaluation against the parent SFT model.

Do not use this model as an authoritative source for mathematics, code, medicine, law, finance, or
other high-stakes decisions. Verify answers independently.

## Limitations and risks

- The evaluation shows a large instruction-following regression; prompts with format constraints
  may be ignored more often than by the parent.
- Generated reasoning traces can be fluent but incorrect, self-contradictory, or fabricated. A
  visible chain of thought is not evidence that the answer is correct.
- The model may switch languages, especially inside long reasoning traces, and European-language
  coverage is uneven.
- Safety behavior was protected only indirectly through 15% exact SFT replay and was not separately
  validated after this stage.
- Synthetic teacher traces dominate the reasoning mixture and can transfer teacher errors, style,
  verbosity, and biases.
- No preference-alignment or deployment hardening was performed after reasoning SFT.

## License and attribution

The model weights are released under Apache-2.0, inherited from the parent model. Training datasets
retain their own licenses and attribution requirements; see the training-data table and linked source
cards. This card does not re-license any dataset.
