"""Configuration loading shared by data, training, inference, UI, and evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a SceneArchitect YAML file or override is invalid."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_FINGERPRINT_IGNORED_PATHS = (
    "checkpointing.resume_from",
    "checkpointing.auto_resume_if_available",
    "run.output_dir",
    "training.max_train_steps",
    "logging.level",
)


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name, default)
        if resolved is None:
            raise ConfigError(f"Environment variable {name!r} is required but not set")
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ConfigError(f"Override must use dotted.path=value syntax: {raw!r}")
    dotted_key, raw_value = raw.split("=", 1)
    keys = dotted_key.split(".")
    if not dotted_key or any(not key or key.startswith("_") for key in keys):
        raise ConfigError(f"Invalid override key: {dotted_key!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML value in override {raw!r}: {exc}") from exc
    return keys, value


def _set_nested(config: MutableMapping[str, Any], keys: list[str], value: Any) -> None:
    current = config
    for key in keys[:-1]:
        existing = current.get(key)
        if existing is None:
            child: dict[str, Any] = {}
            current[key] = child
            current = child
        elif isinstance(existing, MutableMapping):
            current = existing
        else:
            raise ConfigError(f"Cannot set nested key below non-mapping value: {'.'.join(keys)}")
    current[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load YAML, expand ${VAR:-default}, and apply dotted command-line overrides."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Unable to read YAML configuration {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")

    config = _expand_environment(loaded)
    for raw_override in overrides:
        keys, value = _parse_override(raw_override)
        _set_nested(config, keys, _expand_environment(value))

    schema_version = config.get("schema_version")
    if schema_version != 1:
        raise ConfigError(f"Unsupported schema_version={schema_version!r}; expected 1")
    config["_meta"] = {"config_path": str(config_path)}
    return config


def _require(config: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = config
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise ConfigError(f"Missing required configuration key: {dotted_path}")
        current = current[key]
    return current


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate invariants needed before building any large model."""

    required = (
        "run.name",
        "run.target",
        "run.output_dir",
        "model.pretrained_model_name_or_path",
        "data.train_metadata",
        "data.resolution",
        "lora.rank",
        "training.max_train_steps",
        "training.train_batch_size",
        "training.gradient_accumulation_steps",
        "training.mixed_precision",
        "checkpointing.every_n_steps",
        "checkpointing.keep_last",
    )
    for dotted_path in required:
        _require(config, dotted_path)

    target = _require(config, "run.target")
    allowed_targets = {
        "architecture_style",
        "landscape_style",
        "atmosphere_style",
        "character_style",
    }
    if target not in allowed_targets:
        raise ConfigError(f"run.target must be one of {sorted(allowed_targets)}, got {target!r}")

    for key in (
        "data.resolution",
        "lora.rank",
        "training.max_train_steps",
        "training.train_batch_size",
        "training.gradient_accumulation_steps",
        "checkpointing.every_n_steps",
        "checkpointing.keep_last",
    ):
        value = _require(config, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{key} must be a positive integer, got {value!r}")

    resolution = _require(config, "data.resolution")
    if resolution % 8:
        raise ConfigError("data.resolution must be divisible by 8 for the SDXL VAE")
    precision = _require(config, "training.mixed_precision")
    if precision not in {"no", "fp16", "bf16"}:
        raise ConfigError("training.mixed_precision must be one of: no, fp16, bf16")
    if bool(config.get("model", {}).get("train_text_encoders", False)) and resolution >= 768:
        raise ConfigError(
            "The 16GB baseline does not permit text-encoder training at >=768. "
            "Create a separately measured low-resolution experiment instead."
        )


def _remove_dotted_path(config: MutableMapping[str, Any], dotted_path: str) -> None:
    keys = dotted_path.split(".")
    current: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, MutableMapping):
            return
        current = child
    current.pop(keys[-1], None)


def config_fingerprint(
    config: Mapping[str, Any], ignored_paths: Iterable[str] = _FINGERPRINT_IGNORED_PATHS
) -> str:
    """Return a stable hash used to reject accidental cross-experiment resume."""

    clean = copy.deepcopy(dict(config))
    clean.pop("_meta", None)
    for dotted_path in ignored_paths:
        _remove_dotted_path(clean, dotted_path)
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_config(config: Mapping[str, Any]) -> str:
    """Serialize a loaded configuration for inspection without Python tags."""

    return yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False)
