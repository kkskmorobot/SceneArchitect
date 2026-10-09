"""Deterministic dataset validation and control-image preprocessing."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from scenearchitect.config import ConfigError

LOGGER = logging.getLogger(__name__)
_WORD_SEPARATOR = re.compile(r"[_\-.]+")
_SPACE = re.compile(r"\s+")


class DataPreparationError(RuntimeError):
    """Raised when the input dataset cannot produce a valid training split."""


@dataclass(frozen=True)
class PreparationSummary:
    discovered: int
    accepted: int
    rejected: int
    duplicates: int
    train_items: int
    validation_items: int
    output_root: str
    report_path: str


def _resolve_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return (repository_root / path).resolve() if not path.is_absolute() else path.resolve()


def _path_reference(path: Path, repository_root: Path) -> str:
    return (
        path.relative_to(repository_root).as_posix()
        if path.is_relative_to(repository_root)
        else str(path)
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_caption(text: str) -> str:
    text = _WORD_SEPARATOR.sub(" ", text.strip())
    return _SPACE.sub(" ", text).strip(" ,")


def _filename_caption(path: Path, prefix: str) -> str:
    stem = re.sub(r"(?:^|[_-])\d+$", "", path.stem)
    description = _normalise_caption(stem)
    return ", ".join(part for part in (prefix.strip(), description) if part)


class Captioner:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.strategy = str(config.get("strategy", "sidecar_or_filename"))
        self.sidecar_extension = str(config.get("sidecar_extension", ".txt"))
        self.fallback_prefix = str(config.get("fallback_prefix", "anime scene"))
        self.model_name = str(config.get("blip_model", "Salesforce/blip-image-captioning-base"))
        self.device = str(config.get("device", "auto"))
        self._pipeline: Any | None = None
        allowed = {"sidecar", "filename", "sidecar_or_filename", "blip"}
        if self.strategy not in allowed:
            raise ConfigError(f"caption.strategy must be one of {sorted(allowed)}")

    def caption(self, image_path: Path, image: Image.Image) -> str:
        sidecar = image_path.with_suffix(self.sidecar_extension)
        if self.strategy in {"sidecar", "sidecar_or_filename"} and sidecar.is_file():
            caption = _normalise_caption(sidecar.read_text(encoding="utf-8-sig"))
            if caption:
                return caption
        if self.strategy == "sidecar":
            raise DataPreparationError(f"Missing or empty caption sidecar: {sidecar}")
        if self.strategy == "blip":
            return self._caption_with_blip(image)
        return _filename_caption(image_path, self.fallback_prefix)

    def _caption_with_blip(self, image: Image.Image) -> str:
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise DataPreparationError(
                    "BLIP captioning requires the train/data optional dependencies"
                ) from exc
            device = (
                0
                if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())
                else -1
            )
            LOGGER.info("Loading caption model %s on device %s", self.model_name, device)
            self._pipeline = pipeline("image-to-text", model=self.model_name, device=device)
        result = self._pipeline(image)
        if not result or not result[0].get("generated_text"):
            raise DataPreparationError("Caption model returned no text")
        return _normalise_caption(str(result[0]["generated_text"]))


def choose_bucket(width: int, height: int, config: Mapping[str, Any]) -> tuple[int, int]:
    """Choose a step-aligned bucket near the source aspect ratio and pixel budget."""

    step = int(config.get("step", 64))
    minimum = int(config.get("min_side", 512))
    maximum = int(config.get("max_side", 1024))
    max_pixels = int(config.get("max_pixels", 768 * 768))
    if min(step, minimum, maximum, max_pixels, width, height) <= 0 or minimum > maximum:
        raise ConfigError("Invalid bucketing dimensions")

    candidates: list[tuple[int, int]] = []
    for bucket_width in range(minimum, maximum + 1, step):
        for bucket_height in range(minimum, maximum + 1, step):
            if bucket_width * bucket_height <= max_pixels:
                candidates.append((bucket_width, bucket_height))
    if not candidates:
        raise ConfigError("No bucket satisfies bucketing.max_pixels")
    source_ratio = width / height
    return min(
        candidates,
        key=lambda item: (abs(math.log((item[0] / item[1]) / source_ratio)), -item[0] * item[1]),
    )


def _infer_labels(caption: str, labels: Mapping[str, Any]) -> dict[str, str | None]:
    lowered = caption.casefold().replace(" ", "_")
    result: dict[str, str | None] = {}
    for key in ("scene_types", "architecture_styles", "weather", "time_of_day", "camera"):
        values = labels.get(key, [])
        match = next((str(value) for value in values if str(value).casefold() in lowered), None)
        result[key[:-1] if key.endswith("s") else key] = match
    return result


def _save_rgb(image: Image.Image, destination: Path, output: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}{destination.suffix}")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(temporary, quality=int(output.get("jpeg_quality", 95)), optimize=True)
    else:
        image.save(temporary)
    os.replace(temporary, destination)


def _make_canny(image: Image.Image, config: Mapping[str, Any]) -> Image.Image:
    try:
        import cv2
    except ImportError as exc:
        raise DataPreparationError("Canny preprocessing requires opencv-python-headless") from exc
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray,
        threshold1=int(config.get("low_threshold", 100)),
        threshold2=int(config.get("high_threshold", 200)),
    )
    return Image.fromarray(edges, mode="L")


def _make_lineart(image: Image.Image, config: Mapping[str, Any]) -> Image.Image:
    gray = image.convert("L")
    radius = float(config.get("blur_radius", 1.2))
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=max(0.0, radius)))
    array = np.asarray(gray, dtype=np.int16)
    smooth = np.asarray(blurred, dtype=np.int16)
    line = np.clip(255 - np.abs(array - smooth) * 8, 0, 255).astype(np.uint8)
    return Image.fromarray(line, mode="L").filter(ImageFilter.FIND_EDGES)


class OptionalControlProcessor:
    def __init__(self, task: str, config: Mapping[str, Any]) -> None:
        self.task = task
        self.model = str(config.get("model", ""))
        self.device = str(config.get("device", "auto"))
        self._pipeline: Any | None = None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise DataPreparationError(f"{self.task} requires transformers and torch") from exc
            device = (
                0
                if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())
                else -1
            )
            LOGGER.info("Loading %s model %s on device %s", self.task, self.model, device)
            self._pipeline = pipeline(self.task, model=self.model, device=device)
        result = self._pipeline(image)
        if self.task == "depth-estimation":
            depth = result.get("depth")
            if not isinstance(depth, Image.Image):
                raise DataPreparationError("Depth model returned no PIL depth image")
            return ImageOps.autocontrast(depth.convert("L"))
        segmentation = result
        if not isinstance(segmentation, list):
            raise DataPreparationError("Segmentation model returned an unexpected result")
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for index, item in enumerate(segmentation, start=1):
            item_mask = item.get("mask")
            if isinstance(item_mask, Image.Image):
                mask[np.asarray(item_mask.resize(image.size).convert("L")) > 127] = index % 255
        return Image.fromarray(mask, mode="L")


def _discover_images(root: Path, recursive: bool, extensions: Iterable[str]) -> list[Path]:
    allowed = {str(extension).lower() for extension in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in allowed)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _atomic_write_text(path, payload)


def prepare_dataset(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
    dry_run: bool = False,
    limit: int | None = None,
) -> PreparationSummary:
    """Validate raw images and publish deterministic processed metadata and controls."""

    root = Path(repository_root).resolve()
    input_config = config.get("input", {})
    output_config = config.get("output", {})
    input_root = _resolve_path(input_config.get("images_dir", "data/raw/images"), root)
    output_root = _resolve_path(output_config.get("root", "data/processed"), root)
    if not input_root.is_dir():
        raise DataPreparationError(
            f"Input image directory does not exist: {input_root}. Put licensed images there first."
        )
    images = _discover_images(
        input_root,
        bool(input_config.get("recursive", True)),
        input_config.get("extensions", [".jpg", ".jpeg", ".png", ".webp"]),
    )
    if limit is not None:
        if limit <= 0:
            raise DataPreparationError("--limit must be positive")
        images = images[:limit]
    if not images:
        raise DataPreparationError(f"No supported images found below {input_root}")

    captioner = Captioner(config.get("caption", {}))
    controls = config.get("controls", {})
    depth_processor = (
        OptionalControlProcessor("depth-estimation", controls.get("depth", {}))
        if controls.get("depth", {}).get("enabled", False)
        else None
    )
    segmentation_processor = (
        OptionalControlProcessor("image-segmentation", controls.get("segmentation", {}))
        if controls.get("segmentation", {}).get("enabled", False)
        else None
    )
    image_extension = ".jpg" if str(output_config.get("image_format", "jpg")) == "jpg" else ".png"
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0

    for index, source in enumerate(images, start=1):
        try:
            digest = _sha256(source)
            if digest in seen_hashes:
                duplicate_count += 1
                rejected.append({"source": str(source), "reason": "duplicate_sha256"})
                continue
            seen_hashes.add(digest)
            with Image.open(source) as opened:
                if bool(input_config.get("reject_animated", True)) and getattr(
                    opened, "is_animated", False
                ):
                    raise DataPreparationError("animated_image")
                image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            if width < int(input_config.get("min_width", 512)) or height < int(
                input_config.get("min_height", 512)
            ):
                raise DataPreparationError(f"too_small:{width}x{height}")
            if width * height > float(input_config.get("max_megapixels", 40)) * 1_000_000:
                raise DataPreparationError(f"too_large:{width}x{height}")
            caption = captioner.caption(source, image)
            bucket_width, bucket_height = choose_bucket(width, height, config.get("bucketing", {}))
            item_id = f"{index:08d}-{digest[:12]}"
            processed_image = output_root / "images" / f"{item_id}{image_extension}"
            control_paths: dict[str, str] = {}
            if not dry_run:
                overwrite = bool(output_config.get("overwrite", False))
                if overwrite or not processed_image.is_file():
                    _save_rgb(image, processed_image, output_config)
                processors: list[tuple[str, Any]] = []
                if controls.get("canny", {}).get("enabled", True):
                    processors.append(
                        ("canny", lambda value: _make_canny(value, controls["canny"]))
                    )
                if controls.get("lineart", {}).get("enabled", True):
                    processors.append(
                        ("lineart", lambda value: _make_lineart(value, controls["lineart"]))
                    )
                if depth_processor is not None:
                    processors.append(("depth", depth_processor))
                if segmentation_processor is not None:
                    processors.append(("seg", segmentation_processor))
                for name, processor in processors:
                    destination = output_root / "controls" / name / f"{item_id}.png"
                    if overwrite or not destination.is_file():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        processor(image).save(destination)
                    control_paths[name] = _path_reference(destination, root)
            record: dict[str, Any] = {
                "id": item_id,
                "source": _path_reference(source, root),
                "image": _path_reference(processed_image, root),
                "sha256": digest,
                "caption": caption,
                "width": width,
                "height": height,
                "bucket_width": bucket_width,
                "bucket_height": bucket_height,
                "controls": control_paths,
            }
            record.update(_infer_labels(caption, config.get("labels", {})))
            accepted.append(record)
        except (OSError, UnicodeError, UnidentifiedImageError, DataPreparationError) as exc:
            rejected.append({"source": str(source), "reason": str(exc)})
            LOGGER.warning("Rejected %s: %s", source, exc)

    if not accepted:
        raise DataPreparationError(
            "All discovered images were rejected; inspect the preparation report"
        )

    split_config = config.get("split", {})
    validation_fraction = float(split_config.get("validation_fraction", 0.1))
    if not 0 <= validation_fraction < 1:
        raise ConfigError("split.validation_fraction must be in [0, 1)")
    shuffled = list(accepted)
    random.Random(int(split_config.get("seed", 42))).shuffle(shuffled)
    if len(shuffled) <= 1 or validation_fraction == 0:
        validation_count = 0
    else:
        minimum_validation = int(split_config.get("minimum_validation_items", 1))
        validation_count = max(minimum_validation, round(len(shuffled) * validation_fraction))
        validation_count = min(validation_count, len(shuffled) - 1)
    validation_ids = {record["id"] for record in shuffled[:validation_count]}
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    for record in accepted:
        record["split"] = "val" if record["id"] in validation_ids else "train"
        (validation_records if record["split"] == "val" else train_records).append(record)

    report_path = output_root / "metadata" / "preparation_report.json"
    summary = PreparationSummary(
        discovered=len(images),
        accepted=len(accepted),
        rejected=len(rejected),
        duplicates=duplicate_count,
        train_items=len(train_records),
        validation_items=len(validation_records),
        output_root=str(output_root),
        report_path=str(report_path),
    )
    if not dry_run:
        _write_jsonl(output_root / "metadata" / "train.jsonl", train_records)
        _write_jsonl(output_root / "metadata" / "val.jsonl", validation_records)
        _atomic_write_text(
            report_path,
            json.dumps(
                {"summary": asdict(summary), "rejected": rejected}, ensure_ascii=False, indent=2
            ),
        )
    return summary
