#!/usr/bin/env python3
"""Collect lm-eval JSON artifacts into auditable JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_file(root: Path, model_id: str, task_id: str) -> Path:
    matches = sorted((root / "results" / model_id / task_id).glob("**/results_*.json"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected one result for model={model_id} task={task_id}; found {len(matches)}"
        )
    return matches[0]


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = config["models"]
    tasks = config["tasks"]
    rows: list[dict[str, Any]] = []
    task_scores: dict[str, dict[str, float]] = {}
    files: list[dict[str, Any]] = []

    for model in models:
        task_scores[model["id"]] = {}
        for task in tasks:
            path = result_file(args.eval_root, model["id"], task["id"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = payload.get("results", {}).get(task["id"])
            if result is None:
                raise SystemExit(f"missing task result {task['id']} in {path}")
            files.append(
                {
                    "model_id": model["id"],
                    "task": task["id"],
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
            primary_values = []
            for metric in task["primary_metrics"]:
                if metric not in result or not isinstance(result[metric], (int, float)):
                    raise SystemExit(f"missing numeric metric {metric} in {path}")
                value = float(result[metric])
                primary_values.append(value)
                stderr_key = metric.replace(",", "_stderr,", 1)
                stderr = result.get(stderr_key)
                rows.append(
                    {
                        "model_id": model["id"],
                        "model_role": model["role"],
                        "task": task["id"],
                        "capability": task["capability"],
                        "metric": metric,
                        "value": value,
                        "stderr": stderr if isinstance(stderr, (int, float)) else None,
                        "num_fewshot": payload.get("n-shot", {}).get(task["id"]),
                        "result_sha256": files[-1]["sha256"],
                    }
                )
            task_scores[model["id"]][task["id"]] = sum(primary_values) / len(primary_values)

    macro = {
        model["id"]: sum(task_scores[model["id"]].values()) / len(tasks) for model in models
    }
    candidates = [model["id"] for model in models if model["role"] == "reasoning_candidate"]
    selected = max(candidates, key=lambda model_id: (macro[model_id], -candidates.index(model_id)))
    baseline = next(model["id"] for model in models if model["role"] == "starting_sft_checkpoint")

    summary = {
        "schema_version": 1,
        "evaluation": config["version"],
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "eval_root": str(args.eval_root),
        "complete": True,
        "baseline": baseline,
        "selected_candidate": selected,
        "selection_rule": "highest unweighted macro mean of each task's configured primary metrics",
        "macro_mean": macro,
        "task_scores": task_scores,
        "rows": rows,
        "result_files": files,
        "not_run": config.get("not_run", []),
    }
    (args.eval_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (args.eval_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Reasoning-v1 checkpoint comparison",
        "",
        "Scores are percentages. Each task column is the mean of its configured primary metrics; "
        "the macro column is the unweighted mean across tasks.",
        "",
        "| Model | " + " | ".join(task["id"] for task in tasks) + " | Macro | Δ vs baseline |",
        "|---|" + "---:|" * (len(tasks) + 3),
    ]
    for model in models:
        model_id = model["id"]
        values = [fmt(task_scores[model_id][task["id"]]) for task in tasks]
        delta = macro[model_id] - macro[baseline]
        lines.append(
            f"| {model_id} | " + " | ".join(values) + f" | {fmt(macro[model_id])} | {100.0 * delta:+.2f} |"
        )
    lines.extend(
        [
            "",
            f"Selected candidate: **{selected}** under the committed macro-mean rule.",
            "",
            "## Not run",
            "",
        ]
    )
    for item in config.get("not_run", []):
        lines.append(f"- `{item['id']}`: {item['reason']}")
    (args.eval_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_candidate": selected, "macro_mean": macro}, sort_keys=True))


if __name__ == "__main__":
    main()
