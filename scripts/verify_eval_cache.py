#!/usr/bin/env python3
"""Load every task in a generated evaluation matrix before allocating GPUs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from lm_eval.tasks import TaskManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()

    with args.matrix.open(encoding="utf-8", newline="") as handle:
        tasks = sorted({row["task"] for row in csv.DictReader(handle, delimiter="\t")})
    if not tasks:
        raise SystemExit("evaluation matrix contains no tasks")

    manager = TaskManager()
    for task in tasks:
        loaded = manager.load_task_or_group(task)
        if not loaded:
            raise SystemExit(f"task loaded no definitions: {task}")
        print(f"OFFLINE_OK task={task} definitions={','.join(sorted(loaded))}", flush=True)


if __name__ == "__main__":
    main()
