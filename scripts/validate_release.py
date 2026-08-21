#!/usr/bin/env python3
"""Validate release architecture, tokenizer, BF16 weights, and optional GPU inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED = {
    "model_type": "qwen3",
    "max_position_embeddings": 262144,
    "vocab_size": 263168,
    "num_hidden_layers": 36,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    for key, value in EXPECTED.items():
        observed = getattr(config, key)
        if observed != value:
            raise SystemExit(f"architecture mismatch {key}: expected {value}, got {observed}")
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(config, "rope_parameters", {}).get("rope_theta")
    if rope_theta != 64000000:
        raise SystemExit(f"architecture mismatch rope_theta: {rope_theta}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.eos_token != "<end_of_turn>" or tokenizer.pad_token != "<pad>":
        raise SystemExit(
            f"token mismatch eos={tokenizer.eos_token!r} pad={tokenizer.pad_token!r}"
        )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "2 + 2?"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if "<start_of_turn>user\n" not in rendered or not rendered.endswith(
        "<start_of_turn>model\n"
    ):
        raise SystemExit(f"chat-template invariant failed: {rendered!r}")

    weight_files = sorted(args.model.glob("model*.safetensors"))
    if not weight_files:
        raise SystemExit("no model safetensors found")
    tensors = 0
    values = 0
    dtypes: set[str] = set()
    for path in weight_files:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                view = checkpoint.get_slice(name)
                tensors += 1
                values += math.prod(view.get_shape())
                dtypes.add(view.get_dtype())
    if dtypes != {"BF16"}:
        raise SystemExit(f"release weights are not uniformly BF16: {sorted(dtypes)}")
    if tensors != 399 or values != 9_101_947_904:
        raise SystemExit(f"weight inventory mismatch tensors={tensors} values={values}")

    generations = []
    finite_logits = None
    if args.gpu_smoke:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            trust_remote_code=True,
        ).eval()
        prompts = [
            "What is 17 multiplied by 6? Explain briefly.",
            "Förklara kort varför himlen ser blå ut.",
            "Löse kurz: Wenn 3x = 21, was ist x?",
        ]
        for index, prompt in enumerate(prompts):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                if index == 0:
                    logits = model(**inputs).logits
                    finite_logits = bool(torch.isfinite(logits).all().item())
                    if not finite_logits:
                        raise SystemExit("GPU smoke produced non-finite logits")
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=64,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            continuation = tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            if not continuation:
                raise SystemExit(f"empty generation for prompt {index}")
            generations.append({"prompt": prompt, "continuation": continuation})

    payload = {
        "status": "passed",
        "model": str(args.model.resolve()),
        "architecture": {**EXPECTED, "rope_theta": rope_theta},
        "weight_files": len(weight_files),
        "tensors": tensors,
        "values": values,
        "dtypes": sorted(dtypes),
        "chat_template": "passed",
        "gpu_smoke": args.gpu_smoke,
        "finite_logits": finite_logits,
        "generations": generations,
    }
    output = args.output or args.model / "validation.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
