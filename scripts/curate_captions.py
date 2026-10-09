"""Create scene-focused, auditable metadata without modifying raw caption sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import REPOSITORY_ROOT  # noqa: F401

from scenearchitect.config import ConfigError, load_config
from scenearchitect.data.caption_curation import CaptionCurationError, curate_captions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate scene-focused training captions")
    parser.add_argument("--config", default="configs/data/caption_curation.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        config_path = Path(args.config)
        if not config_path.is_absolute() and not config_path.is_file():
            config_path = REPOSITORY_ROOT / config_path
        config = load_config(config_path, args.overrides)
        summary = curate_captions(config, REPOSITORY_ROOT)
    except (ConfigError, CaptionCurationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
