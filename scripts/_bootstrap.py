"""Allow repository scripts to run before an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def run_future_stage(stage: str, feature: str) -> int:
    from scenearchitect.stage_gate import StageNotImplementedError, require_stage

    try:
        require_stage(stage, feature)
    except StageNotImplementedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0
