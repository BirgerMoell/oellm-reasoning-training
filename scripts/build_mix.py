#!/usr/bin/env python3
"""Build a deterministic, token-budgeted reasoning SFT Parquet and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "oellm_gemma_assistant_mask.jinja"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def normalized_prompt(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user":
            return " ".join(message["content"].casefold().split())
    return ""


def clean_messages(raw: Any) -> tuple[list[dict[str, str]] | None, str]:
    if not isinstance(raw, list):
        return None, "messages_not_list"
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None, "message_not_mapping"
        role = str(item.get("role", "")).strip()
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            return None, "unsupported_role"
        if not isinstance(content, str) or not content.strip():
            return None, "empty_content"
        messages.append({"role": role, "content": content.strip()})
    if not any(item["role"] == "user" for item in messages):
        return None, "missing_user"
    if not any(item["role"] == "assistant" for item in messages):
        return None, "missing_assistant"
    if messages[-1]["role"] != "assistant":
        return None, "assistant_not_last"
    return messages, "ok"


def openr1_messages(example: dict[str, Any]) -> tuple[list[dict[str, str]] | None, str]:
    completions = list(example.get("generations") or [])
    complete = list(example.get("is_reasoning_complete") or [])
    math_ok = list(example.get("correctness_math_verify") or [])
    llama_ok = list(example.get("correctness_llama") or [])
    valid_indices = [
        index
        for index in range(len(completions))
        if index < len(complete)
        and bool(complete[index])
        and ((index < len(math_ok) and bool(math_ok[index])) or (index < len(llama_ok) and bool(llama_ok[index])))
    ]
    if int(example.get("correctness_count") or 0) < 1 or not valid_indices:
        return None, "openr1_unverified"

    raw = example.get("messages")
    messages, reason = clean_messages(raw)
    if messages is None:
        prompt = example.get("problem") or example.get("input")
        if not isinstance(prompt, str) or not prompt.strip():
            return None, reason
        messages = [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": str(completions[valid_indices[0]]).strip()},
        ]
    else:
        selected = str(completions[valid_indices[0]]).strip()
        if selected:
            messages[-1] = {"role": "assistant", "content": selected}
    return messages, "ok"


def quotas(config: dict[str, Any]) -> list[int]:
    target = int(config["target_tokens"])
    result = []
    allocated = 0
    for index, source in enumerate(config["sources"]):
        if index == len(config["sources"]) - 1:
            quota = target - allocated
        else:
            quota = int(round(target * float(source["token_share"])))
            allocated += quota
        result.append(quota)
    return result


def resolve_files(source: dict[str, Any], root: Path) -> list[Path]:
    spec = source["input"]
    if spec["kind"] == "local_parquet":
        files = [Path(spec["path"])]
    elif spec["kind"] == "huggingface_snapshot":
        snapshot = root / "raw" / "datasets" / safe_name(spec["repo_id"])
        files = sorted(snapshot.glob(spec["files"]))
        manifest = snapshot / "snapshot.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing snapshot manifest for {source['id']}: {manifest}")
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded.get("revision") != spec["revision"]:
            raise ValueError(
                f"snapshot revision mismatch for {source['id']}: "
                f"{recorded.get('revision')} != {spec['revision']}"
            )
    else:
        raise ValueError(f"unknown input kind for {source['id']}: {spec['kind']}")
    missing = [path for path in files if not path.is_file()]
    if not files or missing:
        raise FileNotFoundError(f"missing input files for {source['id']}: {missing or spec}")
    return files


def normalize_dataset(
    dataset: Dataset,
    source: dict[str, Any],
    tokenizer: Any,
    min_tokens: int,
    max_tokens: int,
    num_proc: int,
) -> tuple[Dataset, Counter[str]]:
    original_columns = list(dataset.column_names)
    adapter = source["adapter"]

    def normalize(example: dict[str, Any]) -> dict[str, Any]:
        if adapter == "openr1_verified":
            messages, reason = openr1_messages(example)
        elif adapter == "messages":
            messages, reason = clean_messages(example.get("messages"))
        else:
            messages, reason = None, "unknown_adapter"

        if messages is None:
            return {
                "messages": [],
                "prompt_hash": "",
                "token_count": 0,
                "source_id": source["id"],
                "language": source["language"],
                "task": source["role"],
                "has_think_tags": False,
                "_valid": False,
                "_reason": reason,
            }

        assistant = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
        if source.get("require_reasoning_trace") and len(assistant) < 128:
            reason = "reasoning_trace_too_short"
            token_count = 0
        else:
            try:
                token_count = len(
                    tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False
                    )
                )
                reason = "ok"
            except Exception:
                token_count = 0
                reason = "template_error"

        if reason == "ok" and token_count < min_tokens:
            reason = "too_short"
        if reason == "ok" and token_count > max_tokens:
            reason = "too_long"
        prompt = normalized_prompt(messages)
        if reason == "ok" and not prompt:
            reason = "missing_normalized_prompt"
        return {
            "messages": messages,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else "",
            "token_count": token_count,
            "source_id": source["id"],
            "language": source["language"],
            "task": source["role"],
            "has_think_tags": "<think>" in assistant and "</think>" in assistant,
            "_valid": reason == "ok",
            "_reason": reason,
        }

    mapped = dataset.map(
        normalize,
        remove_columns=original_columns,
        num_proc=num_proc,
        desc=f"normalize {source['id']}",
    )
    reasons = Counter(mapped["_reason"])
    valid = mapped.filter(
        lambda valid: valid,
        input_columns=["_valid"],
        num_proc=num_proc,
        desc=f"filter {source['id']}",
    ).remove_columns(["_valid", "_reason"])
    return valid, reasons


def select_unique_to_quota(
    dataset: Dataset,
    source_id: str,
    quota: int,
    connection: sqlite3.Connection,
) -> tuple[Dataset, dict[str, int]]:
    selected_indices: list[int] = []
    selected_tokens = 0
    duplicates = 0
    cursor = connection.cursor()
    for index, row in enumerate(dataset):
        cursor.execute(
            "INSERT OR IGNORE INTO prompts(prompt_hash, source_id) VALUES (?, ?)",
            (row["prompt_hash"], source_id),
        )
        if cursor.rowcount == 0:
            duplicates += 1
            continue
        selected_indices.append(index)
        selected_tokens += int(row["token_count"])
        if len(selected_indices) % 5000 == 0:
            connection.commit()
        if selected_tokens >= quota:
            break
    connection.commit()
    if selected_tokens < quota:
        raise RuntimeError(
            f"{source_id} exhausted at {selected_tokens:,} tokens, below quota {quota:,}"
        )
    selected = dataset.select(selected_indices)
    return selected, {
        "selected_rows": len(selected_indices),
        "selected_tokens": selected_tokens,
        "duplicates_skipped": duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-proc", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    output_dir = args.output or (args.root / "data" / config["version"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train.parquet"
    manifest_file = output_dir / "manifest.json"
    database_file = output_dir / "dedup.sqlite3"
    existing = [p for p in (output_file, manifest_file, database_file) if p.exists()]
    if existing and not args.force:
        parser.error(f"output exists; use --force to rebuild: {', '.join(map(str, existing))}")
    if args.force:
        for path in existing:
            path.unlink()

    model_dir = args.root / "models" / config["model"]["local_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.chat_template = TEMPLATE.read_text(encoding="utf-8")

    connection = sqlite3.connect(database_file)
    connection.execute(
        "CREATE TABLE prompts(prompt_hash TEXT PRIMARY KEY, source_id TEXT NOT NULL) WITHOUT ROWID"
    )
    selected_parts: list[Dataset] = []
    source_manifests = []
    per_source_quotas = quotas(config)

    for source_index, (source, quota) in enumerate(zip(config["sources"], per_source_quotas)):
        files = resolve_files(source, args.root)
        print(f"[load] {source['id']} from {len(files)} file(s)", flush=True)
        dataset = load_dataset(
            "parquet", data_files={"train": [str(path) for path in files]}, split="train"
        )
        expected_rows = source["input"].get("expected_rows")
        if expected_rows is not None and len(dataset) != int(expected_rows):
            raise ValueError(
                f"row mismatch for {source['id']}: {len(dataset):,} != {int(expected_rows):,}"
            )
        raw_rows = len(dataset)
        normalized, reasons = normalize_dataset(
            dataset,
            source,
            tokenizer,
            int(config["min_tokens_per_example"]),
            int(config["max_tokens_per_example"]),
            args.num_proc,
        )
        normalized = normalized.shuffle(seed=int(config["seed"]) + source_index)
        selected, selection = select_unique_to_quota(
            normalized, source["id"], quota, connection
        )
        selected_parts.append(selected)
        think_rows = sum(bool(value) for value in selected["has_think_tags"])
        source_manifest = {
            "id": source["id"],
            "role": source["role"],
            "language": source["language"],
            "license": source["license"],
            "requested_token_share": source["token_share"],
            "token_quota": quota,
            "raw_rows": raw_rows,
            "valid_rows": len(normalized),
            "filter_reasons": dict(sorted(reasons.items())),
            "think_tag_rows": think_rows,
            "input": source["input"],
            "resolved_files": [
                {"path": str(path), "bytes": path.stat().st_size} for path in files
            ],
            **selection,
        }
        source_manifests.append(source_manifest)
        print(
            f"[select] {source['id']}: {selection['selected_rows']:,} rows, "
            f"{selection['selected_tokens']:,}/{quota:,} tokens",
            flush=True,
        )

    connection.close()
    combined = concatenate_datasets(selected_parts).shuffle(seed=int(config["seed"]))
    combined.to_parquet(output_file)
    total_tokens = sum(int(item["selected_tokens"]) for item in source_manifests)
    total_rows = len(combined)
    for item in source_manifests:
        item["actual_token_share"] = item["selected_tokens"] / total_tokens

    manifest = {
        "recipe": config["version"],
        "recipe_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "repository_git_sha": git_sha(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_host": os.uname().nodename,
        "seed": config["seed"],
        "model": config["model"],
        "chat_template_sha256": sha256_file(TEMPLATE),
        "target_tokens": config["target_tokens"],
        "selected_tokens": total_tokens,
        "selected_rows": total_rows,
        "min_tokens_per_example": config["min_tokens_per_example"],
        "max_tokens_per_example": config["max_tokens_per_example"],
        "sources": source_manifests,
        "output": {
            "path": str(output_file),
            "bytes": output_file.stat().st_size,
            "sha256": sha256_file(output_file),
        },
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[done] {total_rows:,} rows, {total_tokens:,} tokens, sha256={manifest['output']['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
