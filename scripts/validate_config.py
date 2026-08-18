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
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate() -> list[str]:
    errors: list[str] = []
    data = load(DATA_CONFIG)
    sanity_data = load(SANITY_DATA_CONFIG)
    train = load(TRAIN_CONFIG)
    smoke = load(SMOKE_CONFIG)
    sanity_train = load(SANITY_TRAIN_CONFIG)
    sources = data["sources"]

    weighted = [source for source in sources if source.get("selection") == "token_weighted"]
    consume_once = [source for source in sources if source.get("selection") == "all_once"]
    shares = [float(source["token_share"]) for source in weighted]
    if abs(sum(shares) - 1.0) > 1e-9:
        errors.append(f"token shares sum to {sum(shares)}, not 1.0")
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        errors.append("source IDs are not unique")
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
    if not train["assistant_only_loss"] or not smoke["assistant_only_loss"]:
        errors.append("assistant-only loss must be enabled")
    if train["save_only_model"]:
        errors.append("production checkpoints must include optimizer state")
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
    if not (ROOT / "templates" / "oellm_gemma_assistant_mask.jinja").is_file():
        errors.append("assistant-mask template is missing")
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
