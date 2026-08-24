#!/usr/bin/env python3
"""Measure exact-text repetition in model outputs and a stratified SFT sample.

The strict loop definition follows Pipis et al. (2025): a response is looping
when any 30-gram occurs at least 20 times. Shorter-span diagnostics are also
reported because they catch emerging repetition before it reaches that bar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
STRICT_NGRAM = 30
STRICT_COUNT = 20
SHORT_NGRAM = 8
SHORT_COUNT = 4


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def ngram_counter(tokens: list[str], size: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < size:
        return Counter()
    return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


@dataclass(frozen=True)
class TextMetrics:
    normalized_tokens: int
    repetition_4: float
    max_8gram_count: int
    max_8gram: str | None
    max_30gram_count: int
    strict_loop_30gram_20x: bool
    short_repetition_8gram_4x: bool
    think_tag_status: str


def think_tag_status(text: str) -> str:
    start = text.find("<think>")
    end = text.find("</think>")
    if start == -1 and end == -1:
        return "no_think_tags"
    if start == -1:
        return "closing_tag_only"
    if end == -1 or end < start:
        return "unclosed_think"
    reasoning = text[start + len("<think>") : end]
    return "empty_think" if not reasoning.strip() else "complete_think"


def analyze_text(text: str) -> TextMetrics:
    tokens = normalized_tokens(text)
    grams4 = ngram_counter(tokens, 4)
    total4 = max(0, len(tokens) - 4 + 1)
    repetition4 = 0.0 if total4 == 0 else 1.0 - (len(grams4) / total4)
    grams8 = ngram_counter(tokens, SHORT_NGRAM)
    grams30 = ngram_counter(tokens, STRICT_NGRAM)
    max8, max8_count = (None, 0)
    if grams8:
        max8_tuple, max8_count = max(grams8.items(), key=lambda item: item[1])
        max8 = " ".join(max8_tuple)
    max30_count = max(grams30.values(), default=0)
    return TextMetrics(
        normalized_tokens=len(tokens),
        repetition_4=repetition4,
        max_8gram_count=max8_count,
        max_8gram=max8,
        max_30gram_count=max30_count,
        strict_loop_30gram_20x=max30_count >= STRICT_COUNT,
        short_repetition_8gram_4x=max8_count >= SHORT_COUNT,
        think_tag_status=think_tag_status(text),
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    metrics = [row["metrics"] for row in rows]
    statuses = Counter(metric.think_tag_status for metric in metrics)
    examples = sorted(
        rows,
        key=lambda row: (
            row["metrics"].max_30gram_count,
            row["metrics"].max_8gram_count,
            row["metrics"].repetition_4,
        ),
        reverse=True,
    )[:5]
    return {
        "sampled_rows": len(rows),
        "strict_loop_rows": sum(metric.strict_loop_30gram_20x for metric in metrics),
        "strict_loop_rate": (
            sum(metric.strict_loop_30gram_20x for metric in metrics) / len(rows) if rows else None
        ),
        "short_repetition_rows": sum(metric.short_repetition_8gram_4x for metric in metrics),
        "short_repetition_rate": (
            sum(metric.short_repetition_8gram_4x for metric in metrics) / len(rows) if rows else None
        ),
        "repetition_4_mean": (
            sum(metric.repetition_4 for metric in metrics) / len(metrics) if metrics else None
        ),
        "repetition_4_p95": percentile([metric.repetition_4 for metric in metrics], 0.95),
        "normalized_tokens_mean": (
            sum(metric.normalized_tokens for metric in metrics) / len(metrics) if metrics else None
        ),
        "max_8gram_count": max((metric.max_8gram_count for metric in metrics), default=0),
        "max_30gram_count": max((metric.max_30gram_count for metric in metrics), default=0),
        "think_tag_status": dict(sorted(statuses.items())),
        "highest_repetition_examples": [
            {
                **{key: value for key, value in row.items() if key != "metrics"},
                "metrics": asdict(row["metrics"]),
            }
            for row in examples
        ],
    }


def assistant_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def audit_examples(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for example in document["examples"]:
        metrics = analyze_text(str(example["raw_output"]))
        records.append(
            {
                "id": example["id"],
                "language": example["language"],
                "generated_tokens": example["generated_tokens"],
                "stopped_on_eos": example["stopped_on_eos"],
                "metrics": metrics,
            }
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "summary": summarize(records),
        "examples": [
            {**{key: value for key, value in row.items() if key != "metrics"}, "metrics": asdict(row["metrics"])}
            for row in records
        ],
    }


def audit_parquet(path: Path, manifest_path: Path, max_per_source_language: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(path)
    sample_counts: Counter[tuple[str, str]] = Counter()
    total_counts: Counter[tuple[str, str]] = Counter()
    records: list[dict[str, Any]] = []
    global_index = 0

    columns = ["messages", "source_id", "language", "prompt_hash", "token_count"]
    for batch in parquet.iter_batches(batch_size=1024, columns=columns):
        sources = batch.column("source_id").to_pylist()
        languages = batch.column("language").to_pylist()
        hashes = batch.column("prompt_hash").to_pylist()
        token_counts = batch.column("token_count").to_pylist()
        messages = batch.column("messages")
        for index, (source, language) in enumerate(zip(sources, languages)):
            key = (str(source), str(language))
            total_counts[key] += 1
            if sample_counts[key] < max_per_source_language:
                text = assistant_text(messages[index].as_py())
                records.append(
                    {
                        "source_id": key[0],
                        "language": key[1],
                        "row_index": global_index,
                        "prompt_hash": hashes[index],
                        "rendered_token_count": token_counts[index],
                        "metrics": analyze_text(text),
                    }
                )
                sample_counts[key] += 1
            global_index += 1

    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_language: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source_id"]].append(record)
        by_source_language[(record["source_id"], record["language"])].append(record)

    return {
        "path": str(path),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "recipe": manifest.get("recipe"),
            "selected_rows": manifest.get("selected_rows"),
            "selected_tokens": manifest.get("selected_tokens"),
            "parquet_sha256_from_manifest": manifest.get("output", {}).get("sha256"),
        },
        "sampling": {
            "method": "first N rows per (source_id, language) in the deterministically shuffled Parquet",
            "max_per_source_language": max_per_source_language,
            "sampled_rows": len(records),
            "total_rows_scanned": global_index,
        },
        "overall": summarize(records),
        "by_source": {source: summarize(rows) for source, rows in sorted(by_source.items())},
        "by_source_language": {
            f"{source}:{language}": {
                "population_rows": total_counts[(source, language)],
                **summarize(rows),
            }
            for (source, language), rows in sorted(by_source_language.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, help="model-card examples JSON")
    parser.add_argument("--parquet", type=Path, help="materialized SFT train Parquet")
    parser.add_argument("--manifest", type=Path, help="manifest belonging to --parquet")
    parser.add_argument("--max-per-source-language", type=int, default=2_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.examples and not args.parquet:
        parser.error("provide --examples, --parquet, or both")
    if args.parquet and not args.manifest:
        parser.error("--manifest is required with --parquet")
    if args.max_per_source_language <= 0:
        parser.error("--max-per-source-language must be positive")

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "strict_loop": "any normalized 30-gram occurs at least 20 times",
            "short_repetition_warning": "any normalized 8-gram occurs at least 4 times",
            "repetition_4": "1 - unique normalized 4-grams / total normalized 4-grams",
        },
    }
    if args.examples:
        report["model_examples"] = audit_examples(args.examples)
    if args.parquet:
        report["training_data"] = audit_parquet(
            args.parquet, args.manifest, args.max_per_source_language
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sections": sorted(report)}, indent=2))


if __name__ == "__main__":
    main()
