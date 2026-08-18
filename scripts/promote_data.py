#!/usr/bin/env python3
"""Promote a fully validated isolated data build through an atomic symlink."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-repository-sha", required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    train_file = source / "train.parquet"
    manifest_file = source / "manifest.json"
    dedup_file = source / "dedup.sqlite3"
    for path in (train_file, manifest_file, dedup_file):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty validated artifact file: {path}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("repository_git_sha") != args.expected_repository_sha:
        raise SystemExit(
            "Repository SHA mismatch: "
            f"{manifest.get('repository_git_sha')} != {args.expected_repository_sha}"
        )
    if int(manifest.get("selected_tokens", 0)) < int(manifest.get("target_tokens", 0)):
        raise SystemExit("Manifest selected token count is below target")
    output = manifest.get("output") or {}
    if Path(output.get("path", "")).resolve() != train_file.resolve():
        raise SystemExit("Manifest output path does not identify the promoted Parquet")
    if int(output.get("bytes", -1)) != train_file.stat().st_size:
        raise SystemExit("Manifest output byte count differs from the Parquet")
    if len(str(output.get("sha256", ""))) != 64:
        raise SystemExit("Manifest has no valid-looking Parquet SHA-256")

    destination = args.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"Refusing existing promotion destination: {destination}")
    os.symlink(source, destination, target_is_directory=True)
    if destination.resolve(strict=True) != source:
        raise SystemExit("Promotion symlink did not resolve to the validated source")
    print(
        f"PROMOTED source={source} destination={destination} "
        f"tokens={manifest['selected_tokens']} sha256={output['sha256']}"
    )


if __name__ == "__main__":
    main()
