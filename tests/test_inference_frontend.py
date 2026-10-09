from __future__ import annotations

from pathlib import Path

from PIL import Image

from scenearchitect.config import load_config
from scenearchitect.inference.generator import (
    SceneArchitectGenerator,
    inspect_assets,
    prepare_control_image,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_CONFIG = REPOSITORY_ROOT / "BuShu/frontend_config.yaml"


def test_frontend_output_directory_is_dedicated_frontend_folder() -> None:
    config = load_config(FRONTEND_CONFIG)
    assert config["output"]["directory"] == "BuShu/output"


def test_deployed_profiles_resolve_five_complete_models() -> None:
    config = load_config(FRONTEND_CONFIG)
    assert [item["label"] for item in config["profiles"].values()] == [
        "v1",
        "v2-人物",
        "v2-建筑",
        "v3-人物",
        "v3-建筑",
    ]
    status = inspect_assets(config, REPOSITORY_ROOT)
    for profile in config["profiles"].values():
        model_key = profile["model"]
        adapter_name = profile["adapter"]
        assert status.adapters[model_key][adapter_name] is not None


def test_asset_discovery_prefers_first_complete_adapter(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "weights.safetensors").write_bytes(b"incomplete")
    (second / "weights.safetensors").write_bytes(b"complete")
    (second / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    config = {
        "models": {
            "sdxl": {
                "base_model": "local/base",
                "adapters": {"architecture": {"candidates": [str(first), str(second)]}},
            },
            "animagine": {"base_model": "local/anime", "adapters": {}},
        },
        "controlnets": {
            "canny": {"enabled": True, "model_id": "local/canny"},
            "lineart": {"enabled": False, "model_id": None},
        },
    }
    status = inspect_assets(config, tmp_path)
    assert status.models == {"sdxl": "local/base", "animagine": "local/anime"}
    assert status.adapters["sdxl"]["architecture"] == str(second.resolve())
    assert status.controlnets == {"canny": "local/canny"}


def test_canny_control_preparation_is_rgb_and_requested_size() -> None:
    sketch = Image.new("RGB", (96, 64), "white")
    prepared = prepare_control_image(sketch, "canny", 512, 640)
    assert prepared.size == (512, 640)
    assert prepared.mode == "RGB"


def test_canny_full_uses_the_same_edge_preprocessor() -> None:
    sketch = Image.new("RGB", (96, 64), "white")
    prepared = prepare_control_image(sketch, "canny-full", 512, 640)
    assert prepared.size == (512, 640)
    assert prepared.mode == "RGB"


def test_frontend_exposes_small_and_full_controlnets() -> None:
    config = load_config(FRONTEND_CONFIG)
    assert list(config["controlnets"])[:3] == ["canny", "depth", "canny-full"]
    assert config["controlnets"]["canny-full"]["model_id"] == (
        "models/controlnet-canny-sdxl-1.0"
    )


def test_animagine_prompt_suffix_deduplicates_existing_tags(tmp_path: Path) -> None:
    config = {
        "models": {
            "animagine": {
                "base_model": "local/anime",
                "prompt_suffix": "safe, masterpiece, high score, great score, absurdres",
                "adapters": {},
            }
        },
        "controlnets": {},
    }
    generator = SceneArchitectGenerator(config, tmp_path)
    prompt = generator._prompt_with_prefixes(
        "animagine", "1girl, safe, masterpiece", {}
    )
    assert prompt.count("safe") == 1
    assert prompt.count("masterpiece") == 1
    assert prompt.endswith("high score, great score, absurdres")
