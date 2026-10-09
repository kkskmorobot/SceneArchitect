"""PyCharm-friendly launcher for the local SceneArchitect frontend."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> int:
    deployment_root = Path(__file__).resolve().parent
    root = deployment_root.parent
    python = root / ".venv" / "Scripts" / "python.exe"
    launcher = deployment_root / "launch_ui.py"
    if not python.is_file():
        raise SystemExit(f"Project interpreter not found: {python}")
    if not launcher.is_file():
        raise SystemExit(f"Frontend launcher not found: {launcher}")

    environment = os.environ.copy()
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return subprocess.call(
        [str(python), str(launcher)],
        cwd=root,
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
