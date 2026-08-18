#!/usr/bin/env python3
"""Load the pinned tokenizer locally, including its tokenizers-backend fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast


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
        tokenizer_file = model_path / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise
        print(
            f"[tokenizer] AutoTokenizer failed ({type(error).__name__}); "
            "loading the pinned tokenizer.json backend",
            flush=True,
        )
        return PreTrainedTokenizerFast.from_pretrained(model_path, local_files_only=True)
