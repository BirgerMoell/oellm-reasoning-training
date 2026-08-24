# OELLM 9B 256K Reasoning v1 release

Result: **published publicly** on Hugging Face as
[`birgermoell/oellm-9b-256k-reasoning-v1`](https://huggingface.co/birgermoell/oellm-9b-256k-reasoning-v1).
This is the experimental step-2,000 checkpoint from production training job `21366870`; it is not
presented as an accepted upgrade over the parent SFT model.

## Publication

| Field | Value |
|---|---|
| Published | 2026-08-24 |
| Hugging Face repository | `birgermoell/oellm-9b-256k-reasoning-v1` |
| Hugging Face commit | `6ba6a6aa108ac8aa7a8b31ee634b331c80a9921c` |
| Visibility | public, ungated |
| Remote files | 19 |
| Remote bytes | 18,241,819,669 |
| Released checkpoint | `reasoning-v1/checkpoint-2000` |
| Parent | `birgermoell/oellm-9b-256k-sft@08359ad61333263c067edaf290067fea5b103d34` |
| Weight format | four unquantized BF16 safetensors shards |
| License | Apache-2.0 weights; training datasets retain their upstream terms |

The repository includes the Transformers configuration and tokenizer, detailed model card,
training/data/evaluation YAML, training run record, export manifest, validation report, and a
machine-readable publication record. Publication used `scripts/publish_hf.py`, which refuses a
release without a passed GPU validation report and verifies the authenticated Hugging Face namespace.

## Release artifact

LUMI directory:

```text
/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts/releases/oellm-9b-256k-reasoning-v1
```

| File | Bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00004.safetensors` | 4,999,646,088 | `dfe914562ab5c305119cf5659f8ec0729b25d8c9ad918c5e3616e4c7f7391f86` |
| `model-00002-of-00004.safetensors` | 4,915,951,584 | `f1674363f85efcff3af2f5f8e610e7258a863d5c8301b25d8e22229a754a4412` |
| `model-00003-of-00004.safetensors` | 4,915,960,368 | `8a88a0ddbee4db30aa05a53b309b08ba2b333f7370ab082f5c4a5b8a69c4af80` |
| `model-00004-of-00004.safetensors` | 3,372,383,800 | `f228f0022e4d0d18b38ed9e483578be36d7c0ddd97f4f7971d8b558b9a504f19` |

The export retained `Qwen3ForCausalLM`, 36 layers, vocabulary 263,168, maximum position
configuration 262,144, and RoPE theta 64,000,000. The model card distinguishes this architectural
limit from demonstrated reasoning length because this stage trained at 16,384 tokens.

## Validation

Release-validation job `21492473` completed with exit `0:0` on a LUMI-G GPU. It independently checked:

- uniform BF16 tensors and the expected four-shard index;
- exactly 399 tensors and 9,101,947,904 values;
- every stored value finite: `FULL_FINITE ... nonfinite=0`;
- architecture, tokenizer IDs, and the native chat template;
- a real GPU forward pass with finite logits;
- non-empty greedy continuations for English, Swedish, and German prompts.

The original full-precision step-2,000 checkpoint was separately scanned by job `21426582`, which
also found zero non-finite values across the same 399 tensors and 9,101,947,904 values.

## Verbatim example generation

LUMI job `21492664` completed with exit `0:0` and generated five fixed, independently decoded
English, Swedish, German, French, and Spanish reasoning examples from the published BF16 artifact.
It used the native chat template, greedy decoding, seed `20260824`, and at most 768 new tokens. The
unabridged artifact is committed as `model-card/examples.json` with SHA-256
`890cac7bcf2048f0db852ba8a356a2ec4809a959d5e264365d5523a1c99f4c37`; the model card includes a
manual correctness assessment and three complete verbatim outputs. The examples expose both a
correct English calculation and the looping/unit-reasoning failures consistent with the benchmark
regressions, rather than presenting only favorable generations.

## Evaluation interpretation

The release model card contains the complete parent-versus-candidate table and protocol. The measured
step-2,000 checkpoint improved ARC-Challenge normalized accuracy by 1.79 points and flexible-answer
MGSM in German, Spanish, and French, but regressed GSM8K, IFEval, and MMLU college computer science.
It is therefore published as an experimental research artifact and follow-on post-training base,
not as a production-retained replacement for the parent SFT checkpoint. The extended candidate
MATH-500 rerun was still pending at publication and is explicitly marked as such in the card.
