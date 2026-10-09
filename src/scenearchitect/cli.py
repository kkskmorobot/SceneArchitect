"""Dependency-light SceneArchitect command line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scenearchitect.config import ConfigError, dump_config, load_config
from scenearchitect.runtime import collect_environment, format_environment
from scenearchitect.stage_gate import STAGE_STATUS
from scenearchitect.training.checkpointing import CheckpointError, CheckpointManager


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scene-architect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect Python, CUDA, and optional dependencies")
    doctor.add_argument("--config", default="configs/project.yaml")

    show = subparsers.add_parser("show-config", help="Load and print an expanded YAML file")
    show.add_argument("config")
    show.add_argument("--set", dest="overrides", action="append", default=[])

    checkpoints = subparsers.add_parser("checkpoints", help="Inspect complete training checkpoints")
    checkpoints.add_argument("root")
    action = checkpoints.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--resolve", metavar="REFERENCE")

    subparsers.add_parser("stage-status", help="Print implementation status for phases A-G")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(format_environment(collect_environment(args.config)))
            return 0
        if args.command == "show-config":
            print(dump_config(load_config(args.config, args.overrides)), end="")
            return 0
        if args.command == "checkpoints":
            manager = CheckpointManager(Path(args.root))
            if args.list:
                payload = [record.to_dict() for record in manager.list_complete()]
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                resolved = manager.resolve(args.resolve)
                print(str(resolved) if resolved else "none")
            return 0
        if args.command == "stage-status":
            for stage, status in STAGE_STATUS.items():
                print(f"{stage}: {status}")
            return 0
    except (ConfigError, CheckpointError) as exc:
        parser.error(str(exc))
    return 2
