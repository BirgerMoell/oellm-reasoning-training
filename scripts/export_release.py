#!/usr/bin/env python3
"""Export a consolidated training checkpoint as a compact BF16 HF release."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing existing release directory: {args.output}")
    required = [args.checkpoint / "config.json", args.checkpoint / "tokenizer_config.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if not list(args.checkpoint.glob("model*.safetensors")):
        missing.append(f"{args.checkpoint}/model*.safetensors")
    if missing:
        raise SystemExit(f"missing checkpoint files: {missing}")

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.config.torch_dtype = torch.bfloat16
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=True
    )

    args.output.mkdir(parents=True)
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output)

    weight_files = sorted(args.output.glob("model*.safetensors"))
    payload = {
        "schema_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_step": args.source_step,
        "dtype": "bfloat16",
        "max_shard_size": args.max_shard_size,
        "weights": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in weight_files
        ],
        "config_sha256": sha256_file(args.output / "config.json"),
        "tokenizer_sha256": sha256_file(args.output / "tokenizer.json"),
    }
    (args.output / "export_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
