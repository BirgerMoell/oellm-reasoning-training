#!/usr/bin/env python3
"""Validate staged model invariants and a materialized reasoning-v1 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def validate_model(model_dir: Path, expected: dict) -> None:
    config_file = model_dir / "config.json"
    tokenizer_file = model_dir / "tokenizer_config.json"
    template_file = model_dir / "chat_template.jinja"
    for path in (config_file, tokenizer_file, template_file):
        if not path.is_file():
            fail(f"missing model file: {path}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    architecture = (config.get("architectures") or [None])[0]
    rope = config.get("rope_parameters") or {}
    checks = {
        "architecture": architecture,
        "model_type": config.get("model_type"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "rope_theta": rope.get("rope_theta", config.get("rope_theta")),
        "vocab_size": config.get("vocab_size"),
        "eos_token": tokenizer.get("eos_token"),
        "pad_token": tokenizer.get("pad_token"),
    }
    mismatches = {
        key: (checks.get(key), value)
        for key, value in expected.items()
        if checks.get(key) != value
    }
    if mismatches:
        fail(f"model invariants mismatch: {mismatches}")
    template = template_file.read_text(encoding="utf-8")
    if "<start_of_turn>" not in template or "<end_of_turn>" not in template:
        fail("model chat template is not Gemma-style")
    print(f"OK model: {model_dir}")


def validate_data(root: Path, recipe: dict, full: bool) -> None:
    artifact_dir = root / "data" / recipe["version"]
    parquet_file = artifact_dir / "train.parquet"
    manifest_file = artifact_dir / "manifest.json"
    for path in (parquet_file, manifest_file):
        if not path.is_file():
            fail(f"missing data artifact: {path}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("recipe") != recipe["version"]:
        fail("manifest recipe mismatch")
    if manifest.get("target_tokens") != recipe["target_tokens"]:
        fail("manifest target-token mismatch")
    actual_sha = sha256_file(parquet_file)
    if actual_sha != manifest["output"]["sha256"]:
        fail("Parquet SHA-256 does not match manifest")
    parquet = pq.ParquetFile(parquet_file)
    if parquet.metadata.num_rows != manifest["selected_rows"]:
        fail("Parquet row count does not match manifest")

    source_tokens = {item["id"]: int(item["selected_tokens"]) for item in manifest["sources"]}
    total = sum(source_tokens.values())
    if total < int(recipe["target_tokens"]):
        fail("selected token count is below target")
    expected_ids = {source["id"] for source in recipe["sources"]}
    if set(source_tokens) != expected_ids:
        fail("manifest source IDs differ from recipe")
    weighted_sources = [
        source for source in recipe["sources"] if source["selection"] == "token_weighted"
    ]
    weighted_ids = {source["id"] for source in weighted_sources}
    weighted_total = sum(source_tokens[source_id] for source_id in weighted_ids)
    recipe_shares = {source["id"]: float(source["token_share"]) for source in weighted_sources}
    manifest_by_id = {item["id"]: item for item in manifest["sources"]}
    for source in recipe["sources"]:
        source_id = source["id"]
        if source["selection"] == "all_once":
            expected_rows = source.get("expected_selected_rows")
            if expected_rows is not None and manifest_by_id[source_id]["selected_rows"] != expected_rows:
                fail(f"{source_id} did not include every expected row")
            expected_tokens = source.get("expected_selected_tokens")
            if expected_tokens is not None and source_tokens[source_id] != expected_tokens:
                fail(f"{source_id} rendered token count differs from the recipe")
            expected_languages = source.get("expected_languages")
            if expected_languages is not None and len(manifest_by_id[source_id]["language_counts"]) != expected_languages:
                fail(f"{source_id} language coverage differs from the recipe")
            for reason, expected_count in source.get("expected_filter_reasons", {}).items():
                actual_count = manifest_by_id[source_id]["filter_reasons"].get(reason, 0)
                if actual_count != expected_count:
                    fail(f"{source_id} filter count for {reason} differs from the recipe")
            continue
        actual = source_tokens[source_id] / weighted_total
        if abs(actual - recipe_shares[source_id]) > 0.0025:
            fail(f"{source_id} weighted token share {actual:.4f} is outside tolerance")

    columns = set(parquet.schema_arrow.names)
    required = {"messages", "prompt_hash", "token_count", "source_id", "language", "task"}
    if not required <= columns:
        fail(f"Parquet missing columns: {sorted(required - columns)}")
    first = next(parquet.iter_batches(batch_size=16)).to_pylist()
    for row in first:
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            fail("sample row has no final assistant message")
        if not (recipe["min_tokens_per_example"] <= row["token_count"] <= recipe["max_tokens_per_example"]):
            fail("sample row token count outside recipe bounds")

    if full:
        seen: set[str] = set()
        source_counts: Counter[str] = Counter()
        for batch in parquet.iter_batches(columns=["prompt_hash", "source_id"], batch_size=65536):
            rows = batch.to_pydict()
            for prompt_hash, source_id in zip(rows["prompt_hash"], rows["source_id"]):
                if prompt_hash in seen:
                    fail(f"duplicate prompt hash remains: {prompt_hash}")
                seen.add(prompt_hash)
                source_counts[source_id] += 1
        print(f"OK full dedup scan: {len(seen):,} unique prompts; {dict(source_counts)}")
    print(
        f"OK data: {parquet.metadata.num_rows:,} rows, {total:,} rendered tokens, sha256={actual_sha}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--full", action="store_true", help="scan every prompt hash for duplicates")
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_dir = args.model or (args.root / "models" / recipe["model"]["local_name"])
    validate_model(model_dir, recipe["model"]["expected"])
    validate_data(args.root, recipe, args.full)


if __name__ == "__main__":
    main()
