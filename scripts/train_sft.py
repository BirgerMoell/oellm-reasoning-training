#!/usr/bin/env python3
"""Text-only TRL SFT entry point with LUMI-safe rank-0 loading and hard invariants."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM
from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_peft_config,
)

from tokenizer_utils import load_local_tokenizer


EXPECTED = {
    "model_type": "qwen3",
    "max_position_embeddings": 262144,
    "rope_theta": 64000000,
    "vocab_size": 263168,
}

EXPECTED_LIGER_CONFIG = {
    "rope": False,
    "cross_entropy": False,
    "fused_linear_cross_entropy": True,
    "rms_norm": False,
    "swiglu": False,
}


def validate_training_stack(training_args: SFTConfig) -> None:
    """Fail before model loading unless the 16K-safe fused loss is exactly configured."""
    if not training_args.use_liger_kernel:
        raise RuntimeError("16K reasoning SFT requires the fused Liger loss")
    if training_args.liger_kernel_config != EXPECTED_LIGER_CONFIG:
        raise RuntimeError(
            "unexpected Liger configuration: "
            f"{training_args.liger_kernel_config!r} != {EXPECTED_LIGER_CONFIG!r}"
        )
    try:
        installed = version("liger-kernel")
    except PackageNotFoundError as error:
        raise RuntimeError("liger-kernel is not installed in the LUMI overlay") from error
    if installed != "0.8.1":
        raise RuntimeError(f"liger-kernel version mismatch: {installed!r} != '0.8.1'")


def validate_architecture(config: object) -> None:
    rope = getattr(config, "rope_parameters", None) or {}
    actual = {
        "model_type": getattr(config, "model_type", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "rope_theta": rope.get("rope_theta", getattr(config, "rope_theta", None)),
        "vocab_size": getattr(config, "vocab_size", None),
    }
    mismatches = {key: (actual[key], value) for key, value in EXPECTED.items() if actual[key] != value}
    if mismatches:
        raise RuntimeError(f"starting model architecture mismatch: {mismatches}")


def load_template(tokenizer: object) -> Path:
    raw = os.environ.get("CHAT_TEMPLATE_FILE", "").strip()
    if not raw:
        raise RuntimeError("CHAT_TEMPLATE_FILE is required for assistant-only reasoning SFT")
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(path)
    tokenizer.chat_template = path.read_text(encoding="utf-8")
    return path


def validate_assistant_mask(tokenizer: object, training_args: SFTConfig) -> None:
    if not training_args.assistant_only_loss:
        raise RuntimeError("assistant_only_loss must remain enabled")
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Compute two plus two."},
            {"role": "assistant", "content": "Two plus two is four."},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = rendered.get("assistant_masks") or rendered.get("assistant_tokens_mask")
    if mask is None or not any(mask) or all(mask):
        raise RuntimeError("chat template did not produce a valid assistant token mask")
    print(f"[template] assistant target tokens: {sum(mask)}/{len(mask)}", flush=True)


def prune_no_split_modules(model: torch.nn.Module) -> None:
    names = getattr(model, "_no_split_modules", None)
    if names:
        present = {type(module).__name__ for module in model.modules()}
        model._no_split_modules = [name for name in names if name in present]


def load_model(model_args: ModelConfig) -> torch.nn.Module:
    from accelerate import PartialState

    state = PartialState()
    dtype_name = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    dtype = dtype_name if dtype_name in (None, "auto") else getattr(torch, dtype_name)
    if os.environ.get("RANK0_META_LOAD") == "1" and not state.is_main_process:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            local_files_only=True,
        )
        validate_architecture(config)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=dtype,
                attn_implementation=model_args.attn_implementation,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            dtype=dtype,
            trust_remote_code=model_args.trust_remote_code,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        validate_architecture(model.config)
    prune_no_split_modules(model)
    model.config.use_cache = False
    return model


def write_training_metadata(
    output_dir: Path,
    config_path: Path,
    template_path: Path,
    model_args: ModelConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_snapshot = Path(model_args.model_name_or_path) / "snapshot.json"
    parent_revision = model_args.model_revision
    if model_snapshot.is_file():
        parent_revision = json.loads(model_snapshot.read_text(encoding="utf-8")).get(
            "revision", parent_revision
        )
    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_file = resolved_config["datasets"][0]["data_files"]["train"]
    data_manifest = Path(train_file).with_name("manifest.json")

    def digest(path: Path) -> str | None:
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    payload = {
        "parent_model": model_args.model_name_or_path,
        "parent_revision": parent_revision,
        "training_config": str(config_path),
        "training_config_sha256": digest(config_path),
        "chat_template": str(template_path),
        "chat_template_sha256": digest(template_path),
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": digest(data_manifest),
        "repository_git_sha": os.environ.get("REPOSITORY_GIT_SHA"),
        "container": os.environ.get("TRAINING_CONTAINER"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodes": os.environ.get("SLURM_NNODES"),
        "world_size": os.environ.get("WORLD_SIZE"),
    }
    (output_dir / "reasoning_training_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(
    script_args: ScriptArguments,
    training_args: SFTConfig,
    model_args: ModelConfig,
    dataset_args: DatasetMixtureConfig,
    config_path: Path,
) -> None:
    validate_training_stack(training_args)
    if model_args.attn_implementation != "flash_attention_2":
        raise RuntimeError("packed reasoning SFT requires flash_attention_2")
    model = load_model(model_args)
    tokenizer = load_local_tokenizer(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token != "<end_of_turn>":
        raise RuntimeError(f"unexpected EOS token: {tokenizer.eos_token!r}")
    template_path = load_template(tokenizer)
    validate_assistant_mask(tokenizer, training_args)

    dataset = get_dataset(dataset_args)
    train = dataset[script_args.dataset_train_split]
    if "messages" not in train.column_names:
        raise RuntimeError("training dataset has no messages column")
    train = train.select_columns(["messages"])

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    resume_raw = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()
    if resume_raw in {"", "0"}:
        resume: bool | str | None = None
    elif resume_raw == "1":
        resume = True
    else:
        resume = resume_raw
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        write_training_metadata(
            Path(training_args.output_dir), config_path, template_path, model_args
        )
    trainer.accelerator.print(f"Training completed: {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig))
    script_args, training_args, model_args, dataset_args, remaining = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    config_argument = next(
        (remaining[index + 1] for index, value in enumerate(remaining[:-1]) if value == "--config"),
        os.environ.get("TRAIN_CONFIG", "unknown"),
    )
    main(script_args, training_args, model_args, dataset_args, Path(config_argument))
