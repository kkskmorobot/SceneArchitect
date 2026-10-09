"""Auditable scene-focused caption cleanup for anime background LoRA data."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scenearchitect.config import ConfigError
from scenearchitect.data.metadata import load_metadata

_PERSON_PATTERN = re.compile(r"^(?:\d+|multiple)\s*(?:girls?|boys?|women?|men?)$")


class CaptionCurationError(RuntimeError):
    """Raised when caption curation cannot produce a usable training split."""


@dataclass(frozen=True)
class CurationSummary:
    discovered: int
    kept: int
    rejected: int
    kept_no_humans: int
    kept_person_scenes: int
    train_items: int
    validation_items: int
    audit_path: str
    report_path: str


def _resolve_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return (repository_root / path).resolve() if not path.is_absolute() else path.resolve()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def split_tags(caption: str) -> list[str]:
    """Return unique, normalised comma-separated tags while preserving order."""

    result: list[str] = []
    seen: set[str] = set()
    for raw_tag in caption.split(","):
        tag = " ".join(raw_tag.strip().casefold().replace("_", " ").split())
        if tag and tag not in seen:
            result.append(tag)
            seen.add(tag)
    return result


def _matching_terms(tags: Sequence[str], terms: Sequence[str]) -> list[str]:
    return [tag for tag in tags if any(term in tag for term in terms)]


def _selection_decision(
    tags: Sequence[str], selection: Mapping[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    mode = str(selection.get("mode", "scene")).casefold()
    if mode not in {"scene", "character"}:
        raise ConfigError("selection.mode must be 'scene' or 'character'")
    scene_terms = [str(item).casefold() for item in selection.get("scene_terms", [])]
    composition_tags = {str(item).casefold() for item in selection.get("composition_tags", [])}
    person_focus_tags = {str(item).casefold() for item in selection.get("person_focus_tags", [])}
    minimum_scene_tags = int(selection.get("minimum_scene_tags", 2))
    if minimum_scene_tags < 1:
        raise ConfigError("selection.minimum_scene_tags must be positive")

    tag_set = set(tags)
    no_humans = "no humans" in tag_set
    has_person = any(_PERSON_PATTERN.fullmatch(tag) for tag in tags)
    scene_matches = _matching_terms(tags, scene_terms)
    composition_matches = sorted(tag_set & composition_tags)
    focus_matches = sorted(tag_set & person_focus_tags)
    blocked_tags = {str(item).casefold() for item in selection.get("blocked_tags", [])}
    try:
        blocked_patterns = [
            re.compile(str(item), re.IGNORECASE) for item in selection.get("blocked_patterns", [])
        ]
    except re.error as exc:
        raise ConfigError(f"Invalid selection.blocked_patterns regular expression: {exc}") from exc
    blocked_matches = sorted(
        tag
        for tag in tags
        if tag in blocked_tags or any(pattern.search(tag) for pattern in blocked_patterns)
    )
    details = {
        "mode": mode,
        "no_humans": no_humans,
        "has_person": has_person,
        "scene_tag_count": len(scene_matches),
        "scene_tags": scene_matches,
        "composition_tags": composition_matches,
        "person_focus_tags": focus_matches,
        "blocked_tags": blocked_matches,
    }

    if blocked_matches:
        return False, "blocked_content", details
    if mode == "character":
        if no_humans:
            return False, "character_no_humans", details
        if not has_person:
            return False, "character_without_person_tag", details
        return True, "person_character", details

    if no_humans and bool(selection.get("keep_no_humans", True)):
        return True, "no_humans", details
    if has_person:
        if focus_matches:
            return False, "person_focus", details
        if len(scene_matches) < minimum_scene_tags:
            return False, "person_insufficient_scene_context", details
        if not composition_matches:
            return False, "person_without_wide_composition", details
        return True, "person_scene_composition", details
    if len(scene_matches) >= minimum_scene_tags:
        return True, "scene_without_person_tag", details
    return False, "insufficient_scene_context", details


def clean_caption(
    tags: Sequence[str], caption_config: Mapping[str, Any], *, has_person: bool
) -> str:
    """Remove character-centric tags and add a stable scene-training prefix."""

    prefix_tags = [str(item).strip().casefold() for item in caption_config.get("prefix_tags", [])]
    drop_exact = {str(item).strip().casefold() for item in caption_config.get("drop_exact", [])}
    try:
        drop_patterns = [
            re.compile(str(item), re.IGNORECASE) for item in caption_config.get("drop_patterns", [])
        ]
    except re.error as exc:
        raise ConfigError(f"Invalid caption.drop_patterns regular expression: {exc}") from exc
    max_tags = int(caption_config.get("max_tags", 50))
    if max_tags < 1:
        raise ConfigError("caption.max_tags must be positive")

    cleaned: list[str] = []

    def append(tag: str) -> None:
        if tag and tag not in cleaned and len(cleaned) < max_tags:
            cleaned.append(tag)

    for tag in prefix_tags:
        append(tag)
    if has_person:
        append(str(caption_config.get("person_scale_tag", "distant human figure")).strip())
    for tag in tags:
        if tag in drop_exact or _PERSON_PATTERN.fullmatch(tag):
            continue
        if any(pattern.search(tag) for pattern in drop_patterns):
            continue
        append(tag)
    return ", ".join(cleaned)


def _jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(record), ensure_ascii=False) + "\n" for record in records)


def curate_captions(config: Mapping[str, Any], repository_root: str | Path) -> CurationSummary:
    root = Path(repository_root).resolve()
    input_config = config.get("input", {})
    output_config = config.get("output", {})
    selection = config.get("selection", {})
    caption_config = config.get("caption", {})
    required_input = ("train_metadata", "validation_metadata")
    required_output = ("train_metadata", "validation_metadata", "audit_metadata", "report")
    if not isinstance(input_config, Mapping) or any(
        key not in input_config for key in required_input
    ):
        raise ConfigError(f"caption curation input requires {required_input}")
    if not isinstance(output_config, Mapping) or any(
        key not in output_config for key in required_output
    ):
        raise ConfigError(f"caption curation output requires {required_output}")

    curated_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    audit: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    kept_no_humans = 0
    kept_person_scenes = 0

    for split, key in (("train", "train_metadata"), ("val", "validation_metadata")):
        records = load_metadata(input_config[key], root)
        for record in records:
            raw_caption = str(record["caption"])
            tags = split_tags(raw_caption)
            keep, reason, details = _selection_decision(tags, selection)
            cleaned_caption = clean_caption(
                tags,
                caption_config,
                has_person=details["has_person"] and not details["no_humans"],
            )
            reasons[reason] += 1
            audit.append(
                {
                    "id": record.get("id"),
                    "split": split,
                    "source": record.get("source"),
                    "image": record["image"],
                    "kept": keep,
                    "reason": reason,
                    "raw_caption": raw_caption,
                    "cleaned_caption": cleaned_caption,
                    **details,
                }
            )
            if not keep:
                continue
            if not cleaned_caption:
                raise CaptionCurationError(
                    f"Caption cleaning produced an empty caption for {record['image']}"
                )
            curated = {key: value for key, value in record.items() if not key.startswith("_")}
            curated["raw_caption"] = raw_caption
            curated["caption"] = cleaned_caption
            curated["caption_policy"] = "scene_focus_v1"
            curated["selection_reason"] = reason
            curated_by_split[split].append(curated)
            kept_no_humans += int(reason == "no_humans")
            kept_person_scenes += int(reason == "person_scene_composition")

    if not curated_by_split["train"] or not curated_by_split["val"]:
        raise CaptionCurationError(
            "Caption curation must retain both training and validation items"
        )

    train_path = _resolve_path(output_config["train_metadata"], root)
    val_path = _resolve_path(output_config["validation_metadata"], root)
    audit_path = _resolve_path(output_config["audit_metadata"], root)
    report_path = _resolve_path(output_config["report"], root)
    _atomic_write_text(train_path, _jsonl(curated_by_split["train"]))
    _atomic_write_text(val_path, _jsonl(curated_by_split["val"]))
    _atomic_write_text(audit_path, _jsonl(audit))
    summary = CurationSummary(
        discovered=len(audit),
        kept=len(curated_by_split["train"]) + len(curated_by_split["val"]),
        rejected=sum(1 for item in audit if not item["kept"]),
        kept_no_humans=kept_no_humans,
        kept_person_scenes=kept_person_scenes,
        train_items=len(curated_by_split["train"]),
        validation_items=len(curated_by_split["val"]),
        audit_path=str(audit_path),
        report_path=str(report_path),
    )
    report = {"summary": asdict(summary), "reasons": dict(sorted(reasons.items()))}
    _atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return summary
