#!/usr/bin/env python3
"""Publish a validated release directory to a Hugging Face model repository."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    required = [
        args.model_dir / "README.md",
        args.model_dir / "config.json",
        args.model_dir / "tokenizer.json",
        args.model_dir / "export_manifest.json",
        args.model_dir / "validation.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    weights = sorted(args.model_dir.glob("model*.safetensors"))
    if not weights:
        missing.append(f"{args.model_dir}/model*.safetensors")
    if missing:
        raise SystemExit(f"refusing incomplete release: {missing}")
    validation = json.loads((args.model_dir / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or not validation.get("gpu_smoke"):
        raise SystemExit("refusing release without a passed GPU validation report")

    api = HfApi()
    identity = api.whoami()
    namespace = args.repo_id.split("/", 1)[0]
    if identity["name"] != namespace:
        raise SystemExit(
            f"authenticated as {identity['name']!r}, cannot publish to namespace {namespace!r}"
        )
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.model_dir,
        num_workers=args.num_workers,
        print_report=True,
    )
    info = api.repo_info(args.repo_id, repo_type="model", files_metadata=True)
    payload = {
        "schema_version": 1,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "url": f"https://huggingface.co/{args.repo_id}",
        "private": info.private,
        "commit_sha_before_publication_record": info.sha,
        "files": [
            {"name": item.rfilename, "size": item.size}
            for item in info.siblings
        ],
    }
    record = args.model_dir / "publication.json"
    record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=record,
        path_in_repo="publication.json",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add publication record",
    )
    final = api.repo_info(args.repo_id, repo_type="model", files_metadata=True)
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "url": f"https://huggingface.co/{args.repo_id}",
                "private": final.private,
                "commit_sha": final.sha,
                "files": len(final.siblings),
                "total_bytes": sum(item.size or 0 for item in final.siblings),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
