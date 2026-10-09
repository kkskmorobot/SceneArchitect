"""Stage-D entry point with a real preflight/dry-run and explicit implementation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import REPOSITORY_ROOT  # noqa: F401

from scenearchitect.config import (
    ConfigError,
    config_fingerprint,
    load_config,
    validate_training_config,
)
from scenearchitect.data.metadata import MetadataError, load_metadata
from scenearchitect.logging_utils import configure_logging
from scenearchitect.training.checkpointing import (
    CheckpointError,
    CheckpointManager,
    GracefulStopRequested,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneArchitect SDXL LoRA training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--resume",
        default=None,
        help="Override checkpointing.resume_from: latest, none, or an explicit directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config and resolve resume state without importing PyTorch "
            "or downloading a model"
        ),
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Temporary run override useful for a measured smoke test",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    overrides = list(args.overrides)
    if args.resume is not None:
        overrides.append(f"checkpointing.resume_from={json.dumps(args.resume)}")
    if args.max_train_steps is not None:
        overrides.append(f"training.max_train_steps={args.max_train_steps}")
    try:
        config_path = Path(args.config)
        if not config_path.is_absolute() and not config_path.is_file():
            config_path = REPOSITORY_ROOT / config_path
        config = load_config(config_path, overrides)
        validate_training_config(config)
        configure_logging(str(config.get("logging", {}).get("level", "INFO")))
        checkpointing = config["checkpointing"]
        output_dir = Path(config["run"]["output_dir"])
        if not output_dir.is_absolute():
            output_dir = REPOSITORY_ROOT / output_dir
        output_dir = output_dir.resolve()
        manager = CheckpointManager(output_dir, keep_last=checkpointing["keep_last"])
        resume_reference = checkpointing.get("resume_from")
        resume_path = manager.resolve(resume_reference)
    except (ConfigError, CheckpointError, OSError, ValueError) as exc:
        parser.error(str(exc))

    summary = {
        "config": config["_meta"]["config_path"],
        "run_name": config["run"]["name"],
        "target": config["run"]["target"],
        "output_dir": str(output_dir),
        "config_sha256": config_fingerprint(config),
        "resume_reference": resume_reference,
        "resolved_checkpoint": str(resume_path) if resume_path else None,
        "action": "resume" if resume_path else "start_fresh",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        try:
            records = load_metadata(config["data"]["train_metadata"], REPOSITORY_ROOT)
            print(json.dumps({"train_records": len(records), "data_preflight": "ok"}, indent=2))
        except MetadataError as exc:
            print(
                json.dumps(
                    {"data_preflight": "not_ready", "detail": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0
    try:
        from scenearchitect.training.lora_sdxl import run_training

        result = run_training(config, REPOSITORY_ROOT)
    except (GracefulStopRequested, KeyboardInterrupt) as exc:
        print(f"Training interrupted after saving an emergency checkpoint: {exc}")
        return 130
    except (CheckpointError, MetadataError, RuntimeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
