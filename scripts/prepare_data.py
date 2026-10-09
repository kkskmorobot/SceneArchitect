from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import REPOSITORY_ROOT  # noqa: F401

from scenearchitect.config import ConfigError, load_config
from scenearchitect.data.preprocessing import DataPreparationError, prepare_dataset
from scenearchitect.logging_utils import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SceneArchitect training data")
    parser.add_argument("--config", default="configs/data/preprocess.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing outputs")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        config_path = Path(args.config)
        if not config_path.is_absolute() and not config_path.is_file():
            config_path = REPOSITORY_ROOT / config_path
        config = load_config(config_path, args.overrides)
        configure_logging(str(config.get("logging", {}).get("level", "INFO")))
        summary = prepare_dataset(
            config,
            repository_root=REPOSITORY_ROOT,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    except (ConfigError, DataPreparationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
