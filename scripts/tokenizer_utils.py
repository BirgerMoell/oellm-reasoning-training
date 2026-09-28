#!/usr/bin/env python3
"""Load the pinned tokenizer locally, including its tokenizers-backend fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def _token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def load_tokenizer_json_backend(model_path: Path) -> PreTrainedTokenizerFast:
    """Load tokenizer.json without asking Transformers to reinterpret legacy class metadata."""

    tokenizer_file = model_path / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(tokenizer_file)
    config_file = model_path / "tokenizer_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
    kwargs: dict[str, Any] = {}
    standard_tokens: set[str] = set()
    for name in ("unk_token", "bos_token", "eos_token", "pad_token"):
        token = _token_content(config.get(name))
        if token is not None:
            kwargs[name] = token
            standard_tokens.add(token)
    raw_extras = config.get("additional_special_tokens") or config.get("extra_special_tokens") or []
    if isinstance(raw_extras, dict):
        raw_extras = list(raw_extras.values())
    if isinstance(raw_extras, list):
        extras = [
            token
            for token in (_token_content(value) for value in raw_extras)
            if token is not None and token not in standard_tokens
        ]
        if extras:
            kwargs["additional_special_tokens"] = extras
    model_max_length = config.get("model_max_length")
    if isinstance(model_max_length, (int, float)):
        kwargs["model_max_length"] = int(model_max_length)
    if isinstance(config.get("clean_up_tokenization_spaces"), bool):
        kwargs["clean_up_tokenization_spaces"] = config["clean_up_tokenization_spaces"]
    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **kwargs)


def load_local_tokenizer(model_dir: str | Path, trust_remote_code: bool = False) -> Any:
    """Load through AutoTokenizer, or use tokenizer.json when legacy class metadata is incompatible."""

    model_path = Path(model_dir)
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except (TypeError, ValueError) as error:
        if not (model_path / "tokenizer.json").is_file():
            raise
        print(
            f"[tokenizer] AutoTokenizer failed ({type(error).__name__}); "
            "loading the pinned tokenizer.json backend",
            flush=True,
        )
        return load_tokenizer_json_backend(model_path)
