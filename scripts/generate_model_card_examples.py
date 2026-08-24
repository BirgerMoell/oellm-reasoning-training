#!/usr/bin/env python3
"""Generate deterministic, auditable model-card examples from a local checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def split_reasoning(raw_output: str) -> tuple[str | None, str, str]:
    """Split a complete <think> trace without inventing structure for untagged output."""

    start = raw_output.find("<think>")
    end = raw_output.find("</think>", start + len("<think>")) if start >= 0 else -1
    if start >= 0 and end >= 0:
        reasoning = raw_output[start + len("<think>") : end].strip()
        answer = raw_output[end + len("</think>") :].strip()
        return reasoning, answer, "complete_think_tags"
    if start >= 0:
        return raw_output[start + len("<think>") :].strip(), "", "unclosed_think_tag"
    return None, raw_output.strip(), "no_think_tags"


def main() -> None:
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    generation = config["generation"]
    if generation.get("do_sample") is not False:
        raise SystemExit("model-card examples must use deterministic greedy decoding")
    seed = int(generation["seed"])
    max_new_tokens = int(generation["max_new_tokens"])
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise SystemExit("a GPU is required to generate model-card examples")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=True,
    ).eval()

    examples = []
    for spec in config["examples"]:
        prompt = str(spec["prompt"]).strip()
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[0, prompt_tokens:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not raw_output:
            raise SystemExit(f"empty output for {spec['id']}")
        reasoning, answer, trace_format = split_reasoning(raw_output)
        examples.append(
            {
                **spec,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(generated.shape[0]),
                "stopped_on_eos": bool(generated[-1].item() == tokenizer.eos_token_id),
                "trace_format": trace_format,
                "reasoning_trace": reasoning,
                "answer": answer,
                "raw_output": raw_output,
                "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "model_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "prompt_config": str(args.config),
        "generation": {
            "seed": seed,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "dtype": "bfloat16",
            "device": torch.cuda.get_device_name(0),
        },
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
