#!/usr/bin/env python3
"""Validate a staged model's pinned chat template and assistant loss mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tokenizer_utils import load_local_tokenizer
from validate_run import validate_model


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    recipe = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_spec = recipe["model"]
    model_dir = args.root / "models" / model_spec["local_name"]
    validate_model(model_dir, model_spec["expected"])

    template_path = Path(model_spec["chat_template"])
    if not template_path.is_absolute():
        template_path = ROOT / template_path
    committed_template = template_path.read_text(encoding="utf-8")
    staged_template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if committed_template.rstrip() != staged_template.rstrip():
        raise SystemExit("VALIDATION FAILED: committed template differs from the pinned parent")

    tokenizer = load_local_tokenizer(model_dir)
    tokenizer.chat_template = committed_template
    for name, expected_id in model_spec["special_token_ids"].items():
        token = {
            "turn_start": "<|im_start|>",
            "turn_end": "<|im_end|>",
            "think_end": "</think>",
            "think_start": "<think>",
            "pad": "<pad>",
        }[name]
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise SystemExit(
                f"VALIDATION FAILED: {token} id {actual_id} differs from {expected_id}"
            )

    turn_end = model_spec["expected"]["turn_end_token"]
    turn_end_id = tokenizer.convert_tokens_to_ids(turn_end)
    tokenizer.eos_token = turn_end
    messages = [
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "<think>Two plus two equals four.</think>Four."},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = rendered.get("assistant_masks") or rendered.get("assistant_tokens_mask")
    if mask is None or not any(mask) or all(mask):
        raise SystemExit("VALIDATION FAILED: native template produced an invalid assistant mask")
    supervised_ids = [token for token, enabled in zip(rendered["input_ids"], mask) if enabled]
    if turn_end_id not in supervised_ids:
        raise SystemExit("VALIDATION FAILED: assistant turn terminator is not supervised")
    last_turn_end = len(supervised_ids) - 1 - supervised_ids[::-1].index(turn_end_id)
    if tokenizer.decode(supervised_ids[last_turn_end + 1 :]).strip():
        raise SystemExit("VALIDATION FAILED: targets remain after the assistant turn terminator")

    reasoning_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if "<think>\nTwo plus two equals four.\n</think>\n\nFour.<|im_end|>" not in reasoning_text:
        raise SystemExit("VALIDATION FAILED: reasoning trace was not normalized into the think channel")
    replay_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if "<think>\n\n</think>\n\nHello.<|im_end|>" not in replay_text:
        raise SystemExit("VALIDATION FAILED: replay row does not preserve the empty-think convention")

    print(
        "OK chat template: "
        f"assistant_targets={sum(mask)}/{len(mask)} turn_end_id={turn_end_id} "
        f"template={template_path}"
    )


if __name__ == "__main__":
    main()
