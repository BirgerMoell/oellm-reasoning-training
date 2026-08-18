#!/usr/bin/env python3
"""Verify that saved model weights are finite without loading a whole checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_TENSORS = (
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.20.mlp.down_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("model*.safetensors"))
    else:
        raise SystemExit(f"Checkpoint does not exist: {path}")
    if not files or any(file.suffix != ".safetensors" for file in files):
        raise SystemExit(f"No model safetensors found at: {path}")
    return files


def count_nonfinite(view: object, shape: list[int], max_chunk_values: int) -> int:
    if not shape:
        return int((~torch.isfinite(view[...])).sum())
    trailing_values = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows_per_chunk = max(1, max_chunk_values // trailing_values)
    count = 0
    for start in range(0, shape[0], rows_per_chunk):
        index = (slice(start, min(shape[0], start + rows_per_chunk)),) + tuple(
            slice(None) for _ in shape[1:]
        )
        count += int((~torch.isfinite(view[index])).sum())
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="stream every value in every model shard; otherwise sample key tensors",
    )
    parser.add_argument(
        "--max-chunk-values",
        type=int,
        default=64 * 1024 * 1024,
        help="maximum values materialized at once during a full scan",
    )
    parser.add_argument("--expected-tensors", type=int)
    parser.add_argument("--expected-values", type=int)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    files = checkpoint_files(args.checkpoint)

    locations: dict[str, Path] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name in locations:
                    raise SystemExit(f"Duplicate tensor {name} in {locations[name]} and {file}")
                locations[name] = file

    missing = [name for name in DEFAULT_TENSORS if name not in locations]
    if missing:
        raise SystemExit(f"Required model tensors are missing: {', '.join(missing)}")

    if args.full:
        tensor_count = value_count = nonfinite_count = 0
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as checkpoint:
                for name in sorted(checkpoint.keys()):
                    view = checkpoint.get_slice(name)
                    shape = view.get_shape()
                    tensor_nonfinite = count_nonfinite(view, shape, args.max_chunk_values)
                    tensor_count += 1
                    value_count += math.prod(shape)
                    nonfinite_count += tensor_nonfinite
                    if tensor_nonfinite:
                        print(f"NONFINITE {name} shape={shape} count={tensor_nonfinite}")
        structure_ok = True
        if args.expected_tensors is not None and tensor_count != args.expected_tensors:
            print(f"TENSOR_COUNT_MISMATCH actual={tensor_count} expected={args.expected_tensors}")
            structure_ok = False
        if args.expected_values is not None and value_count != args.expected_values:
            print(f"VALUE_COUNT_MISMATCH actual={value_count} expected={args.expected_values}")
            structure_ok = False
        if nonfinite_count:
            state = "FULL_NONFINITE"
        elif not structure_ok:
            state = "FULL_INVALID"
        else:
            state = "FULL_FINITE"
        print(
            f"{state} files={len(files)} tensors={tensor_count} "
            f"values={value_count} nonfinite={nonfinite_count}"
        )
        raise SystemExit(0 if nonfinite_count == 0 and structure_ok else 1)

    all_finite = True
    for name in DEFAULT_TENSORS:
        file = locations.get(name)
        assert file is not None
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            view = checkpoint.get_slice(name)
            shape = view.get_shape()
            index = tuple(slice(0, min(8, size)) for size in shape)
            sample = view[index].float() if shape else view[...].float()
            finite = bool(torch.isfinite(sample).all())
            all_finite &= finite
            print(
                f"{'FINITE' if finite else 'NONFINITE'} {name} "
                f"shape={shape} sample={sample.numel()} "
                f"nan={int(torch.isnan(sample).sum())} "
                f"inf={int(torch.isinf(sample).sum())} "
                f"absmax={float(torch.nan_to_num(sample).abs().max()):.8g}"
            )
    print(f"{'SAMPLE_FINITE' if all_finite else 'SAMPLE_INVALID'} files={len(files)}")
    raise SystemExit(0 if all_finite else 1)


if __name__ == "__main__":
    main()
