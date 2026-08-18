#!/usr/bin/env python3
"""Stage the exact model and Hugging Face dataset snapshots for offline LUMI jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def file_manifest(root: Path) -> list[dict[str, object]]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts):
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    return files


def write_snapshot_manifest(
    destination: Path, repo_id: str, revision: str, repo_type: str, patterns: list[str]
) -> None:
    payload = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "allow_patterns": patterns,
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "files": file_manifest(destination),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    (destination / "snapshot.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def stage(
    repo_id: str,
    revision: str,
    repo_type: str,
    destination: Path,
    patterns: list[str],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[stage] {repo_type} {repo_id}@{revision} -> {destination}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=destination,
        allow_patterns=patterns,
    )
    write_snapshot_manifest(destination, repo_id, revision, repo_type, patterns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], help="stage only these source IDs")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--datasets-only", action="store_true")
    args = parser.parse_args()

    if args.model_only and args.datasets_only:
        parser.error("--model-only and --datasets-only are mutually exclusive")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selected = set(args.source)
    known_ids = {source["id"] for source in config["sources"]}
    unknown = selected - known_ids
    if unknown:
        parser.error(f"unknown source IDs: {', '.join(sorted(unknown))}")

    if not args.datasets_only:
        model = config["model"]
        stage(
            model["repo_id"],
            model["revision"],
            "model",
            args.root / "models" / model["local_name"],
            ["*.json", "*.jinja", "*.model", "*.safetensors", "*.safetensors.index.json"],
        )

    if args.model_only:
        return

    repositories: dict[tuple[str, str], set[str]] = defaultdict(set)
    repo_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source in config["sources"]:
        if selected and source["id"] not in selected:
            continue
        spec = source["input"]
        if spec["kind"] != "huggingface_snapshot":
            continue
        key = (spec["repo_id"], spec["revision"])
        repositories[key].update({spec["files"], "README.md", ".gitattributes"})
        repo_sources[key].add(source["id"])

    for (repo_id, revision), patterns in repositories.items():
        print(f"[stage] covers source IDs: {', '.join(sorted(repo_sources[(repo_id, revision)]))}")
        stage(
            repo_id,
            revision,
            "dataset",
            args.root / "raw" / "datasets" / safe_name(repo_id),
            sorted(patterns),
        )


if __name__ == "__main__":
    main()
