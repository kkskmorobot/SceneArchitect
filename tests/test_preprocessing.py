from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scenearchitect.data.preprocessing import choose_bucket, prepare_dataset


def _config(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "input": {
            "images_dir": str(root / "raw"),
            "recursive": True,
            "extensions": [".png"],
            "min_width": 32,
            "min_height": 32,
            "max_megapixels": 2,
        },
        "output": {"root": str(root / "processed"), "image_format": "jpg"},
        "caption": {"strategy": "sidecar_or_filename", "fallback_prefix": "anime scene"},
        "labels": {"scene_types": ["forest"], "weather": ["rain"]},
        "controls": {
            "canny": {"enabled": True, "low_threshold": 50, "high_threshold": 100},
            "lineart": {"enabled": True, "blur_radius": 1.0},
            "depth": {"enabled": False},
            "segmentation": {"enabled": False},
        },
        "bucketing": {"step": 32, "min_side": 32, "max_side": 128, "max_pixels": 8192},
        "split": {"validation_fraction": 0.34, "seed": 7, "minimum_validation_items": 1},
    }


def test_choose_bucket_tracks_aspect_ratio() -> None:
    bucket = choose_bucket(
        1024,
        512,
        {"step": 64, "min_side": 256, "max_side": 1024, "max_pixels": 512 * 512},
    )
    assert bucket[0] > bucket[1]
    assert bucket[0] * bucket[1] <= 512 * 512


def test_prepare_dataset_writes_controls_and_deterministic_split(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index, colour in enumerate(("red", "green", "blue")):
        path = raw / f"forest_rain_{index}.png"
        Image.new("RGB", (96 + index * 8, 64), colour).save(path)
    (raw / "forest_rain_0.txt").write_text("forest, rain, wide shot", encoding="utf-8")

    summary = prepare_dataset(_config(tmp_path), repository_root=tmp_path)
    assert summary.accepted == 3
    assert summary.train_items == 2
    assert summary.validation_items == 1

    metadata_root = tmp_path / "processed" / "metadata"
    records = [
        json.loads(line) for line in (metadata_root / "train.jsonl").read_text().splitlines()
    ]
    records += [json.loads(line) for line in (metadata_root / "val.jsonl").read_text().splitlines()]
    assert all((tmp_path / record["image"]).is_file() for record in records)
    assert all((tmp_path / record["controls"]["canny"]).is_file() for record in records)
    assert all((tmp_path / record["controls"]["lineart"]).is_file() for record in records)
    assert any(record["caption"] == "forest, rain, wide shot" for record in records)
