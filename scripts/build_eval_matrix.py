#!/usr/bin/env python3
"""Resolve the committed checkpoint-selection recipe into a LUMI TSV matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def selected(value: str, requested: set[str]) -> bool:
    return not requested or value in requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_filter = set(args.model)
    task_filter = set(args.task)
    known_models = {item["id"] for item in config["models"]}
    known_tasks = {item["id"] for item in config["tasks"]}
    if model_filter - known_models:
        parser.error(f"unknown models: {sorted(model_filter - known_models)}")
    if task_filter - known_tasks:
        parser.error(f"unknown tasks: {sorted(task_filter - known_tasks)}")

    rows: list[dict[str, str | int]] = []
    resolved_models = []
    for model in config["models"]:
        if not selected(model["id"], model_filter):
            continue
        model_path = args.root / model["path"]
        required = [model_path / "config.json", model_path / "tokenizer_config.json"]
        missing = [str(path) for path in required if not path.is_file()]
        weight_files = sorted(model_path.glob("model*.safetensors"))
        if not weight_files:
            missing.append(f"{model_path}/model*.safetensors")
        if missing:
            raise SystemExit(f"missing model files for {model['id']}: {missing}")
        resolved_models.append(
            {
                **model,
                "resolved_path": str(model_path),
                "config_sha256": sha256_file(model_path / "config.json"),
            }
        )
        for task in config["tasks"]:
            if not selected(task["id"], task_filter):
                continue
            rows.append(
                {
                    "model_id": model["id"],
                    "model_path": str(model_path),
                    "task": task["id"],
                    "capability": task["capability"],
                    "num_fewshot": int(task["num_fewshot"]),
                    "max_gen_toks": int(task["max_gen_toks"]),
                }
            )
    if not rows:
        raise SystemExit("evaluation selection produced no rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    payload = {
        "version": config["version"],
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "matrix": str(args.output),
        "matrix_sha256": sha256_file(args.output),
        "root": str(args.root),
        "models": resolved_models,
        "tasks": [
            task for task in config["tasks"] if selected(task["id"], task_filter)
        ],
        "rows": len(rows),
        "harness": config["harness"],
        "seed": config["seed"],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} evaluations to {args.output}")


if __name__ == "__main__":
    main()
