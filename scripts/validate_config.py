#!/usr/bin/env python3
"""Static validation for committed data/training recipes."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "configs" / "data" / "reasoning-v1.yaml"
SANITY_DATA_CONFIG = ROOT / "configs" / "data" / "reasoning-sanity.yaml"
SMOKE_CONFIG = ROOT / "configs" / "train" / "smoke.yaml"
SANITY_TRAIN_CONFIG = ROOT / "configs" / "train" / "sanity.yaml"
TRAIN_CONFIG = ROOT / "configs" / "train" / "reasoning-v1.yaml"
ANNEAL_DATA_CONFIG = ROOT / "configs" / "data" / "reasoning-anneal300b-v1.yaml"
ANNEAL_TRAIN_CONFIG = ROOT / "configs" / "train" / "reasoning-anneal300b-v1.yaml"
ANNEAL_SANITY_CONFIG = ROOT / "configs" / "train" / "reasoning-anneal300b-sanity.yaml"
TRANSLATED_DATA_CONFIG = (
    ROOT / "configs" / "data" / "reasoning-anneal300b-dolci-translated-v2.yaml"
)
TRANSLATED_TRAIN_CONFIG = (
    ROOT / "configs" / "train" / "reasoning-anneal300b-dolci-translated-v2.yaml"
)
TRANSLATED_SANITY_CONFIG = (
    ROOT / "configs" / "train" / "reasoning-anneal300b-dolci-translated-sanity.yaml"
)
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_anneal300b() -> list[str]:
    errors: list[str] = []
    data = load(ANNEAL_DATA_CONFIG)
    train = load(ANNEAL_TRAIN_CONFIG)
    sanity = load(ANNEAL_SANITY_CONFIG)
    sources = data["sources"]
    weighted = [source for source in sources if source["selection"] == "token_weighted"]
    shares = sum(float(source["token_share"]) for source in weighted)
    if abs(shares - 1.0) > 1e-9:
        errors.append(f"anneal300b token shares sum to {shares}, not 1.0")
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        errors.append("anneal300b source IDs are not unique")
    order = data.get("selection_order") or []
    if len(order) != len(set(order)) or set(order) != set(ids):
        errors.append("anneal300b selection_order must contain every source ID exactly once")
    if not order or order[0] != "reasoning-traces-multilingual-v0.2-pilot":
        errors.append("anneal300b multilingual coverage source must select first")
    if not COMMIT.match(data["model"]["revision"]):
        errors.append("anneal300b model revision is not a 40-character commit")
    for source in sources:
        spec = source["input"]
        if spec["kind"] == "huggingface_snapshot" and not COMMIT.match(spec["revision"]):
            errors.append(f"anneal300b {source['id']} revision is not a commit")
        if source.get("require_reasoning_trace"):
            if source.get("reasoning_format") != "think_and_answer":
                errors.append(f"anneal300b {source['id']} does not require think-plus-answer format")
            if not source.get("reject_strict_repetition"):
                errors.append(f"anneal300b {source['id']} does not reject strict lexical loops")
    replay = next((source for source in sources if source["id"] == "dolci-instruct-sft-replay"), None)
    if replay is None or float(replay.get("token_share", 0)) < 0.35:
        errors.append("anneal300b requires at least 35% exact-source instruction replay")
    expected = data["model"]["expected"]
    if expected.get("turn_end_token") != "<|im_end|>" or expected.get("eos_token") != "<eos>":
        errors.append("anneal300b model token invariants changed")
    if data["model"].get("special_token_ids") != {
        "turn_start": 3,
        "turn_end": 4,
        "think_end": 17,
        "think_start": 18,
        "pad": 262144,
    }:
        errors.append("anneal300b special-token IDs changed")
    template = ROOT / data["model"]["chat_template"]
    if not template.is_file():
        errors.append("anneal300b native assistant-mask template is missing")
    elif "<|im_start|>" not in template.read_text(encoding="utf-8"):
        errors.append("anneal300b template is not ChatML-style")
    if train.get("eos_token") != "<|im_end|>" or sanity.get("eos_token") != "<|im_end|>":
        errors.append("anneal300b trainer EOS must be the assistant turn terminator")
    if train["max_steps"] * 64 * train["max_length"] != data["target_tokens"]:
        errors.append("anneal300b packed training budget differs from the data target")
    allowed_sanity_differences = {
        "output_dir",
        "max_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
    }
    train_core = {key: value for key, value in train.items() if key not in allowed_sanity_differences}
    sanity_core = {key: value for key, value in sanity.items() if key not in allowed_sanity_differences}
    if train_core != sanity_core:
        errors.append("anneal300b sanity differs from production outside run-size/output fields")
    if sanity.get("max_steps") != 30:
        errors.append("anneal300b recovery sanity must cross the old stall with 30 steps")
    if not str(sanity.get("output_dir", "")).endswith("reasoning-anneal300b-sanity-v2"):
        errors.append("anneal300b recovery sanity must use a fresh v2 output directory")
    for source_id in ("dolci-instruct-sft-replay",):
        if not (ROOT / "data" / "sources" / source_id / "README.md").is_file():
            errors.append(f"missing source card for {source_id}")
    for wrapper in (
        ROOT / "slurm" / "train_anneal300b_sanity_lumi.sbatch",
        ROOT / "slurm" / "train_anneal300b_production_lumi.sbatch",
    ):
        text = wrapper.read_text(encoding="utf-8") if wrapper.is_file() else ""
        for required in (
            "#SBATCH --nodes=8",
            "#SBATCH --gpus-per-node=8",
            "templates/oellm_qwen3_assistant_mask.jinja",
            "exec bash slurm/train_lumi.sbatch",
        ):
            if required not in text:
                errors.append(f"{wrapper.name} missing: {required}")
    return errors


def validate_translated_anneal300b() -> list[str]:
    """Protect the v2 translated-Dolci experiment and its one-variable run comparison."""
    errors: list[str] = []
    data = load(TRANSLATED_DATA_CONFIG)
    train = load(TRANSLATED_TRAIN_CONFIG)
    sanity = load(TRANSLATED_SANITY_CONFIG)
    sources = data["sources"]
    weighted = [source for source in sources if source["selection"] == "token_weighted"]
    shares = sum(float(source["token_share"]) for source in weighted)
    if abs(shares - 1.0) > 1e-9:
        errors.append(f"translated v2 token shares sum to {shares}, not 1.0")
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        errors.append("translated v2 source IDs are not unique")
    order = data.get("selection_order") or []
    if len(order) != len(set(order)) or set(order) != set(ids):
        errors.append("translated v2 selection_order must contain every source exactly once")

    languages = {"cs", "de", "el", "es", "fi", "fr", "it", "nl", "pl", "ro", "sv", "uk"}
    translated = [source for source in sources if source["id"].startswith("dolci-think-translated-")]
    if {source["language"] for source in translated} != languages:
        errors.append("translated v2 must contain exactly the reviewed 12 languages")
    if len(translated) != 12 or any(float(source["token_share"]) != 0.02 for source in translated):
        errors.append("translated v2 must allocate exactly 2% per translated language")
    for source in translated:
        spec = source["input"]
        if spec.get("repo_id") != "openeurollm/Dolci-Think-SFT-translated":
            errors.append(f"{source['id']} uses the wrong translated repository")
        if spec.get("revision") != "ba4754ab30afb66e652c3690ef0390dcea4939cd":
            errors.append(f"{source['id']} translated revision changed")
        if not source.get("require_reasoning_trace") or not source.get("reject_strict_repetition"):
            errors.append(f"{source['id']} is missing reasoning/repetition gates")
    replay = next((source for source in sources if source["id"] == "dolci-instruct-sft-replay"), None)
    if replay is None or float(replay.get("token_share", 0)) != 0.35:
        errors.append("translated v2 must retain 35% Dolci instruction replay")
    medqa = next((source for source in sources if source["id"] == "medqa-correct-reasoning-traces"), None)
    if medqa is None:
        errors.append("translated v2 is missing the MedQA reasoning slice")
    else:
        expected_medqa = {
            "repo_id": "birgermoell/medqa-reasoning-traces",
            "revision": "82fe04a7165fdafc5bacb7f6a988d4d0763d6d9a",
            "split": "train",
            "files": "data/train-*.parquet",
            "expected_rows": 10178,
        }
        for key, value in expected_medqa.items():
            if medqa["input"].get(key) != value:
                errors.append(f"translated v2 MedQA {key} changed")
        if float(medqa.get("token_share", 0)) != 0.02:
            errors.append("translated v2 MedQA must remain 2% of weighted tokens")
        if medqa.get("adapter") != "medqa_correct_reasoning":
            errors.append("translated v2 MedQA must use the correct/completed-trace adapter")
    if train["max_steps"] * 64 * train["max_length"] != data["target_tokens"]:
        errors.append("translated v2 packed training budget differs from its data target")
    allowed_sanity_differences = {
        "output_dir",
        "max_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
    }
    train_core = {key: value for key, value in train.items() if key not in allowed_sanity_differences}
    sanity_core = {key: value for key, value in sanity.items() if key not in allowed_sanity_differences}
    if train_core != sanity_core:
        errors.append("translated v2 sanity differs from production outside run-size/output fields")
    if sanity.get("max_steps") != 30 or sanity.get("save_steps") != 10:
        errors.append("translated v2 sanity must save every 10 steps through step 30")
    for name, config in (("translated v2 production", train), ("translated v2 sanity", sanity)):
        if config.get("use_liger_kernel") or config.get("loss_type") != "chunked_nll":
            errors.append(f"{name} must use chunked_nll with Liger disabled")
        if int(config.get("ddp_timeout", 0)) > 900:
            errors.append(f"{name} DDP timeout exceeds the 15-minute ceiling")
    if not (ROOT / "data" / "sources" / "dolci-think-sft-translated" / "README.md").is_file():
        errors.append("translated Dolci source card is missing")
    if not (ROOT / "data" / "sources" / "medqa-reasoning-traces" / "README.md").is_file():
        errors.append("MedQA reasoning source card is missing")
    for wrapper in (
        ROOT / "slurm" / "train_anneal300b_dolci_translated_sanity_lumi.sbatch",
        ROOT / "slurm" / "train_anneal300b_dolci_translated_production_lumi.sbatch",
    ):
        text = wrapper.read_text(encoding="utf-8") if wrapper.is_file() else ""
        for required in (
            "#SBATCH --nodes=8",
            "#SBATCH --gpus-per-node=8",
            "templates/oellm_qwen3_assistant_mask.jinja",
            "exec bash slurm/train_lumi.sbatch",
        ):
            if required not in text:
                errors.append(f"{wrapper.name} missing: {required}")
    return errors


def validate() -> list[str]:
    errors: list[str] = []
    data = load(DATA_CONFIG)
    sanity_data = load(SANITY_DATA_CONFIG)
    train = load(TRAIN_CONFIG)
    smoke = load(SMOKE_CONFIG)
    sanity_train = load(SANITY_TRAIN_CONFIG)
    sources = data["sources"]
    selection_order = data.get("selection_order")

    weighted = [source for source in sources if source.get("selection") == "token_weighted"]
    consume_once = [source for source in sources if source.get("selection") == "all_once"]
    shares = [float(source["token_share"]) for source in weighted]
    if abs(sum(shares) - 1.0) > 1e-9:
        errors.append(f"token shares sum to {sum(shares)}, not 1.0")
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        errors.append("source IDs are not unique")
    if selection_order is None:
        errors.append("production selection_order is missing")
    elif len(selection_order) != len(set(selection_order)) or set(selection_order) != set(ids):
        errors.append("selection_order must contain every production source ID exactly once")
    elif selection_order[0] != "reasoning-traces-multilingual-v0.2-pilot":
        errors.append("the fixed multilingual pilot must be first in selection_order")
    if any(share <= 0 for share in shares):
        errors.append("every token-weighted share must be positive")
    if any("token_share" in source for source in consume_once):
        errors.append("consume-once sources must not define token_share")
    required_consume_once = {
        "expected_selected_rows",
        "expected_selected_tokens",
        "expected_languages",
        "expected_filter_reasons",
    }
    for source in consume_once:
        missing = required_consume_once - set(source)
        if missing:
            errors.append(f"{source['id']} consume-once invariants missing: {sorted(missing)}")
    selections = [source.get("selection") for source in sources]
    if any(selection not in {"all_once", "token_weighted"} for selection in selections):
        errors.append("every source must use all_once or token_weighted selection")
    first_weighted = next(
        (index for index, selection in enumerate(selections) if selection == "token_weighted"),
        len(selections),
    )
    if any(selection == "all_once" for selection in selections[first_weighted:]):
        errors.append("consume-once sources must appear before token-weighted sources")

    sanity_sources = sanity_data["sources"]
    sanity_shares = [float(source["token_share"]) for source in sanity_sources]
    if abs(sum(sanity_shares) - 1.0) > 1e-9:
        errors.append(f"sanity token shares sum to {sum(sanity_shares)}, not 1.0")
    if any(source.get("selection") != "token_weighted" for source in sanity_sources):
        errors.append("every sanity source must be token-weighted")
    if any(int(source.get("max_raw_rows", 0)) <= 0 for source in sanity_sources):
        errors.append("every sanity source must define a positive max_raw_rows")
    if any(int(source["max_raw_rows"]) != 5000 for source in sanity_sources):
        errors.append("every sanity source must use the reviewed 5,000-row cap")
    if any(int(source.get("max_input_files", 0)) != 1 for source in sanity_sources):
        errors.append("every sanity source must use exactly one resolved input file")
    if [source["id"] for source in sanity_sources] != ids:
        errors.append("sanity source IDs/order differ from production")
    production_inputs = {source["id"]: source["input"] for source in sources}
    if any(source["input"] != production_inputs.get(source["id"]) for source in sanity_sources):
        errors.append("sanity inputs/revisions differ from production")
    if sanity_data["model"] != data["model"]:
        errors.append("sanity model differs from production")

    if not COMMIT.match(data["model"]["revision"]):
        errors.append("model revision is not a 40-character commit")
    for source in sources:
        spec = source["input"]
        if spec["kind"] == "huggingface_snapshot" and not COMMIT.match(spec["revision"]):
            errors.append(f"{source['id']} revision is not a commit")
        if source.get("require_reasoning_trace") and source["role"] == "capability_replay":
            errors.append(f"{source['id']} replay unexpectedly requires a reasoning trace")
        card = ROOT / "data" / "sources" / source["id"].replace(
            "nemotron-v2-math-decontaminated", "nemotron-post-training-v2"
        )
        # Nemotron slices intentionally share one source card; the remaining configured IDs have cards.
        if source["id"] == "reasoning-traces-multilingual-v0.2-pilot":
            card = ROOT / "data" / "sources" / "reasoning-traces-multilingual"
        elif source["id"].startswith("nemotron-v2-"):
            card = ROOT / "data" / "sources" / "nemotron-post-training-v2"
        elif source["id"] == "openr1-math-220k-verified":
            card = ROOT / "data" / "sources" / "openr1-math-220k"
        if not (card / "README.md").is_file():
            errors.append(f"missing source card for {source['id']}")

    expected = data["model"]["expected"]
    if expected["max_position_embeddings"] != 262144 or expected["rope_theta"] != 64000000:
        errors.append("model architecture invariants changed")
    if train["max_length"] != data["max_tokens_per_example"]:
        errors.append("training max_length does not match data max tokens")
    if sanity_train["max_length"] != sanity_data["max_tokens_per_example"]:
        errors.append("sanity training max_length does not match sanity data max tokens")
    allowed_sanity_differences = {
        "datasets",
        "output_dir",
        "max_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
    }
    production_training_core = {
        key: value for key, value in train.items() if key not in allowed_sanity_differences
    }
    sanity_training_core = {
        key: value for key, value in sanity_train.items() if key not in allowed_sanity_differences
    }
    if sanity_training_core != production_training_core:
        errors.append("sanity training differs from production outside the allowed run-size fields")
    if train["attn_implementation"] != "flash_attention_2" or not train["packing"]:
        errors.append("production packing requires flash_attention_2")
    training_configs = (
        ("production", train),
        ("sanity", sanity_train),
        ("smoke", smoke),
        ("anneal300b production", load(ANNEAL_TRAIN_CONFIG)),
        ("anneal300b sanity", load(ANNEAL_SANITY_CONFIG)),
    )
    for name, config in training_configs:
        if config.get("use_liger_kernel"):
            errors.append(f"{name} training must keep Liger disabled")
        if config.get("loss_type") != "chunked_nll":
            errors.append(f"{name} training must use TRL chunked_nll")
        if int(config.get("ddp_timeout", 0)) > 900:
            errors.append(f"{name} DDP timeout exceeds the 15-minute fail-fast ceiling")
    if not train["assistant_only_loss"] or not smoke["assistant_only_loss"]:
        errors.append("assistant-only loss must be enabled")
    if train["save_only_model"]:
        errors.append("production checkpoints must include optimizer state")
    if train["save_steps"] != 250:
        errors.append("production save interval must remain 250 steps")
    required_checkpoint_count = train["max_steps"] // train["save_steps"]
    if train["save_total_limit"] < required_checkpoint_count:
        errors.append(
            "production checkpoint retention would delete required evaluation/recovery checkpoints"
        )
    if train["max_steps"] != 2000:
        errors.append("production max_steps must match the v1 token budget")
    packed_budget = train["max_steps"] * 64 * train["max_length"]
    if packed_budget != data["target_tokens"]:
        errors.append(f"packed training budget {packed_budget} != data target {data['target_tokens']}")
    sanity_packed_budget = sanity_train["max_steps"] * 64 * sanity_train["max_length"]
    if sanity_packed_budget > sanity_data["target_tokens"]:
        errors.append("sanity data target is smaller than its packed training budget")
    sanity_wrapper = (ROOT / "slurm" / "train_sanity_lumi.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --nodes=8",
        "#SBATCH --gpus-per-node=8",
        "exec bash slurm/train_lumi.sbatch",
    ):
        if required not in sanity_wrapper:
            errors.append(f"sanity Slurm wrapper missing: {required}")
    production_wrapper = (ROOT / "slurm" / "train_production_lumi.sbatch").read_text(
        encoding="utf-8"
    )
    for required in (
        "#SBATCH --nodes=8",
        "#SBATCH --gpus-per-node=8",
        "#SBATCH --time=0-14:00:00",
        "export TRAIN_CONFIG=configs/train/reasoning-v1.yaml",
        "exec bash slurm/train_lumi.sbatch",
    ):
        if required not in production_wrapper:
            errors.append(f"production Slurm wrapper missing: {required}")
    if not (ROOT / "templates" / "oellm_gemma_assistant_mask.jinja").is_file():
        errors.append("assistant-mask template is missing")
    errors.extend(validate_anneal300b())
    errors.extend(validate_translated_anneal300b())
    return errors


def main() -> None:
    errors = validate()
    if errors:
        raise SystemExit("configuration validation failed:\n- " + "\n- ".join(errors))
    data = load(DATA_CONFIG)
    print(
        f"OK: {len(data['sources'])} slices, weighted shares=1.0, "
        f"target={int(data['target_tokens']):,} tokens"
    )


if __name__ == "__main__":
    main()
