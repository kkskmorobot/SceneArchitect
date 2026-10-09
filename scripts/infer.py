from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from _bootstrap import REPOSITORY_ROOT  # noqa: F401
from PIL import Image

from scenearchitect.config import ConfigError, load_config
from scenearchitect.inference.generator import (
    GenerationError,
    SceneArchitectGenerator,
    inspect_assets,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneArchitect SDXL inference")
    parser.add_argument("--config", default="configs/inference/sdxl_controlnet_16gb.yaml")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--base-model", choices=["sdxl", "animagine"], default="sdxl")
    parser.add_argument(
        "--mode", choices=["base", "architecture", "character", "blend"], default="architecture"
    )
    parser.add_argument("--architecture-weight", type=float, default=0.85)
    parser.add_argument("--character-weight", type=float, default=0.8)
    parser.add_argument("--sketch", default=None)
    parser.add_argument("--control-type", default="canny")
    parser.add_argument("--control-scale", type=float, default=0.75)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=6.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument(
        "--inspect", action="store_true", help="Inspect assets without loading Torch"
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        config_path = Path(args.config)
        if not config_path.is_absolute() and not config_path.is_file():
            config_path = REPOSITORY_ROOT / config_path
        config = load_config(config_path)
        if args.inspect:
            print(
                json.dumps(
                    inspect_assets(config, REPOSITORY_ROOT).__dict__, ensure_ascii=False, indent=2
                )
            )
            return 0
        if not args.prompt:
            parser.error("--prompt is required unless --inspect is used")
        weights = {
            "architecture": args.architecture_weight
            if args.mode in {"architecture", "blend"}
            else 0.0,
            "character": args.character_weight if args.mode in {"character", "blend"} else 0.0,
        }
        sketch = Image.open(args.sketch).convert("RGB") if args.sketch else None
        generator = SceneArchitectGenerator(config, REPOSITORY_ROOT)
        result = generator.generate(
            model_key=args.base_model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            adapter_weights=weights,
            control_image=sketch,
            control_type=args.control_type if sketch else None,
            control_scale=args.control_scale,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            num_images=args.num_images,
        )
    except (ConfigError, GenerationError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
