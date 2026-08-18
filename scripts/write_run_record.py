#!/usr/bin/env python3
"""Create or update the small YAML provenance record for a LUMI training attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--status", choices=("started", "completed"), required=True)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.run_dir / "run.yaml"
    record = yaml.safe_load(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = args.root / "data" / "reasoning-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    model_path = Path(config["model_name_or_path"])
    snapshot_path = model_path / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.is_file() else {}
    template_path = Path("templates/oellm_gemma_assistant_mask.jinja")
    repo_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    repo_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    job_id = os.environ.get("SLURM_JOB_ID")
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("NUM_PROC", "1")))

    record.update(
        {
            "run_id": job_id,
            "status": args.status,
            "repository": {
                "url": "https://github.com/BirgerMoell/oellm-reasoning-training",
                "git_sha": repo_sha,
                "dirty": repo_dirty,
            },
            "parent_model": {
                "path": str(model_path),
                "repo_id": snapshot.get("repo_id"),
                "revision": snapshot.get("revision"),
                "config_sha256": sha256_file(model_path / "config.json"),
            },
            "data": {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "parquet": str(args.root / "data" / "reasoning-v1" / "train.parquet"),
                "parquet_sha256": manifest.get("output", {}).get("sha256"),
                "selected_rows": manifest.get("selected_rows"),
                "selected_tokens": manifest.get("selected_tokens"),
            },
            "training": {
                "config": str(args.config),
                "config_sha256": sha256_file(args.config),
                "template": str(template_path),
                "template_sha256": sha256_file(template_path),
                "output_dir": config.get("output_dir"),
            },
            "container": args.container,
        }
    )

    attempts = record.setdefault("attempts", [])
    if not attempts or attempts[-1].get("slurm_job_id") != job_id:
        attempts.append(
            {
                "slurm_job_id": job_id,
                "status": args.status,
                "started_at": utc_now(),
                "nodes": int(os.environ.get("SLURM_NNODES", "1")),
                "world_size": world_size,
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "account": os.environ.get("SLURM_JOB_ACCOUNT"),
                "resume_from_checkpoint": os.environ.get("RESUME_FROM_CHECKPOINT", "0"),
            }
        )
    else:
        attempts[-1]["status"] = args.status
    if args.status == "completed":
        attempts[-1]["completed_at"] = utc_now()

    record_path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    print(f"Run record {args.status}: {record_path}")


if __name__ == "__main__":
    main()
