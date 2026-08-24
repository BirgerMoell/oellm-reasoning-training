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
