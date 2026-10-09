"""Launch the SceneArchitect Gradio frontend from the BuShu deployment folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEPLOYMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = DEPLOYMENT_ROOT.parent
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app import build_app  # noqa: E402

from scenearchitect.config import ConfigError, load_config  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the SceneArchitect frontend")
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = DEPLOYMENT_ROOT / "frontend_config.yaml"
    try:
        config = load_config(config_path)
        demo = build_app(config, REPOSITORY_ROOT)
        ui_config = config.get("ui", {})
        demo.launch(
            server_name=str(ui_config.get("host", "127.0.0.1")),
            server_port=int(ui_config.get("port", 7860)),
            share=False,
            inbrowser=not args.no_browser,
            show_error=True,
        )
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"SceneArchitect frontend startup failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
