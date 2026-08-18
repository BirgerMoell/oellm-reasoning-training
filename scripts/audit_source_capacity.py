#!/usr/bin/env python3
"""Measure eligible unique-token capacity for selected recipe sources."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset

from build_mix import normalize_dataset, resolve_files, weighted_quotas
from tokenizer_utils import load_local_tokenizer


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "oellm_gemma_assistant_mask.jinja"


def seed_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {row[0] for row in connection.execute("SELECT prompt_hash FROM prompts")}
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--seed-dedup", type=Path)
    parser.add_argument("--num-proc", type=int, default=32)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    by_id = {source["id"]: source for source in config["sources"]}
    unknown = [source_id for source_id in args.source if source_id not in by_id]
    if unknown:
        parser.error(f"unknown source IDs: {', '.join(unknown)}")

    fixed_tokens = sum(
        int(source.get("expected_selected_tokens", 0))
        for source in config["sources"]
        if source["selection"] == "all_once"
    )
    quotas = weighted_quotas(config, fixed_tokens)
    model_dir = args.root / "models" / config["model"]["local_name"]
    tokenizer = load_local_tokenizer(model_dir)
    tokenizer.chat_template = TEMPLATE.read_text(encoding="utf-8")
    seen = seed_hashes(args.seed_dedup)
    print(f"[seed] {len(seen):,} prompt hashes", flush=True)

    insufficient = False
    for source_id in args.source:
        source = by_id[source_id]
        files = resolve_files(source, args.root)
        print(f"[load] {source_id} from {len(files)} file(s)", flush=True)
        dataset = load_dataset(
            "parquet", data_files={"train": [str(path) for path in files]}, split="train"
        )
        normalized, reasons = normalize_dataset(
            dataset,
            source,
            tokenizer,
            int(config["min_tokens_per_example"]),
            int(config["max_tokens_per_example"]),
            args.num_proc,
        )
        unique_rows = duplicate_rows = eligible_tokens = 0
        for row in normalized:
            prompt_hash = row["prompt_hash"]
            if prompt_hash in seen:
                duplicate_rows += 1
                continue
            seen.add(prompt_hash)
            unique_rows += 1
            eligible_tokens += int(row["token_count"])
        quota = quotas[source_id]
        enough = eligible_tokens >= quota
        insufficient |= not enough
        print(
            json.dumps(
                {
                    "source_id": source_id,
                    "input_rows": len(dataset),
                    "valid_rows": len(normalized),
                    "unique_rows_after_seed": unique_rows,
                    "duplicates_after_seed": duplicate_rows,
                    "eligible_tokens_after_seed": eligible_tokens,
                    "requested_quota": quota,
                    "headroom_tokens": eligible_tokens - quota,
                    "sufficient": enough,
                    "filter_reasons": dict(sorted(Counter(reasons).items())),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    raise SystemExit(2 if insufficient else 0)


if __name__ == "__main__":
    main()
