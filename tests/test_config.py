from __future__ import annotations

from pathlib import Path

import pytest

from scenearchitect.config import (
    ConfigError,
    config_fingerprint,
    load_config,
    validate_training_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_training_config_loads_and_validates() -> None:
    config = load_config(
        ROOT / "configs/train/sdxl_lora_16gb.yaml",
        ["training.max_train_steps=20", "run.name=smoke_test"],
    )
    validate_training_config(config)
    assert config["training"]["max_train_steps"] == 20
    assert config["run"]["name"] == "smoke_test"


def test_character_training_config_loads_and_validates() -> None:
    config = load_config(ROOT / "configs/train/sdxl_lora_character_v2.yaml")
    validate_training_config(config)
    assert config["run"]["target"] == "character_style"


def test_resume_reference_does_not_change_fingerprint() -> None:
    base = load_config(ROOT / "configs/train/sdxl_lora_16gb.yaml")
    changed = load_config(
        ROOT / "configs/train/sdxl_lora_16gb.yaml", ["checkpointing.resume_from=none"]
    )
    assert config_fingerprint(base) == config_fingerprint(changed)


def test_extending_max_steps_does_not_break_resume_fingerprint() -> None:
    base = load_config(ROOT / "configs/train/sdxl_lora_16gb.yaml")
    extended = load_config(
        ROOT / "configs/train/sdxl_lora_16gb.yaml", ["training.max_train_steps=20000"]
    )
    assert config_fingerprint(base) == config_fingerprint(extended)


def test_text_encoder_training_guard_for_16gb() -> None:
    config = load_config(
        ROOT / "configs/train/sdxl_lora_16gb.yaml", ["model.train_text_encoders=true"]
    )
    with pytest.raises(ConfigError, match="text-encoder training"):
        validate_training_config(config)
