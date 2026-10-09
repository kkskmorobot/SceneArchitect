"""Dependency-light JSONL metadata loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetadataError(ValueError):
    pass


def load_metadata(path: str | Path, repository_root: str | Path) -> list[dict[str, Any]]:
    metadata_path = Path(path).expanduser()
    root = Path(repository_root).resolve()
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    if not metadata_path.is_file():
        raise MetadataError(f"Training metadata does not exist: {metadata_path}")
    records: list[dict[str, Any]] = []
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MetadataError(f"Invalid JSON at {metadata_path}:{line_number}: {exc}") from exc
        for key in ("image", "caption", "bucket_width", "bucket_height"):
            if key not in record:
                raise MetadataError(f"Missing {key!r} at {metadata_path}:{line_number}")
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = root / image_path
        if not image_path.is_file():
            raise MetadataError(f"Image referenced at line {line_number} is missing: {image_path}")
        record["_image_path"] = str(image_path.resolve())
        records.append(record)
    if not records:
        raise MetadataError(f"Training metadata contains no records: {metadata_path}")
    return records
