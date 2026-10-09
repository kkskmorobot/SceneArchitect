"""Atomic Accelerate checkpoint save, discovery, resume, and interruption helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import signal
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any

from scenearchitect.config import config_fingerprint

LOGGER = logging.getLogger(__name__)
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-step-(\d{12})(?:-[a-z0-9-]+)?$")
_METADATA_FILE = "metadata.json"
_MANIFEST_FILE = "manifest.json"
_SUCCESS_FILE = "_SUCCESS"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, incomplete, or incompatible."""


class GracefulStopRequested(KeyboardInterrupt):
    """Raised at a safe training boundary after SIGINT/SIGTERM."""


@dataclass
class TrainingProgress:
    """Loop counters registered with Accelerator for exact resume."""

    global_step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0
    samples_seen: int = 0
    best_metric: float | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key in asdict(self):
            if key in state:
                setattr(self, key, state[key])


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    global_step: int
    reason: str
    created_at: str
    config_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "global_step": self.global_step,
            "reason": self.reason,
            "created_at": self.created_at,
            "config_sha256": self.config_sha256,
        }


class CheckpointManager:
    """Manage complete checkpoints below one experiment output directory."""

    def __init__(self, root: str | Path, keep_last: int = 3) -> None:
        self.root = Path(root).expanduser().resolve()
        if keep_last <= 0:
            raise ValueError("keep_last must be positive")
        self.keep_last = keep_last

    def list_complete(self) -> list[CheckpointRecord]:
        if not self.root.exists():
            return []
        records: list[CheckpointRecord] = []
        for path in self.root.iterdir():
            match = _CHECKPOINT_PATTERN.fullmatch(path.name)
            if not match or not path.is_dir() or not (path / _SUCCESS_FILE).is_file():
                continue
            metadata_path = path / _METADATA_FILE
            try:
                self._validate_files(path, verify_hashes=False)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                step = int(metadata.get("global_step", match.group(1)))
                records.append(
                    CheckpointRecord(
                        path=path.resolve(),
                        global_step=step,
                        reason=str(metadata.get("reason", "unknown")),
                        created_at=str(metadata.get("created_at", "")),
                        config_sha256=metadata.get("config_sha256"),
                    )
                )
            except (CheckpointError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                LOGGER.warning("Ignoring unreadable checkpoint %s: %s", path, exc)
        records.sort(key=lambda item: (item.global_step, item.created_at, item.path.name))
        return records

    def latest(self) -> Path | None:
        records = self.list_complete()
        return records[-1].path if records else None

    def resolve(self, reference: str | Path | None) -> Path | None:
        if reference is None or str(reference).strip().lower() in {"", "none", "false"}:
            return None
        if str(reference).strip().lower() == "latest":
            return self.latest()

        candidate = Path(reference).expanduser()
        if not candidate.is_absolute():
            rooted = self.root / candidate
            candidate = rooted if rooted.exists() else candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise CheckpointError(f"Checkpoint directory does not exist: {candidate}")
        self._validate_files(candidate, verify_hashes=False)
        return candidate

    def should_save(self, global_step: int, every_n_steps: int) -> bool:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        return global_step > 0 and global_step % every_n_steps == 0

    def save(
        self,
        accelerator: Any,
        progress: TrainingProgress,
        config: Mapping[str, Any],
        reason: str = "periodic",
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save Accelerate state to a temp directory, then publish it atomically.

        The project targets one process / one GPU. All Accelerate processes still call
        save_state; only the main process writes SceneArchitect metadata and rotates files.
        """

        safe_reason = re.sub(r"[^a-z0-9]+", "-", reason.lower()).strip("-") or "manual"
        base_name = f"checkpoint-step-{progress.global_step:012d}"
        created_at = datetime.now(timezone.utc).isoformat()
        suffix = ""
        final_path = self.root / base_name
        if final_path.exists():
            suffix = f"-{safe_reason}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            final_path = self.root / f"{base_name}{suffix}"
        temp_path = self.root / f".{base_name}{suffix}-{uuid.uuid4().hex}.tmp"

        if getattr(accelerator, "is_main_process", True):
            self.root.mkdir(parents=True, exist_ok=True)
            temp_path.mkdir(parents=False, exist_ok=False)
        if hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

        try:
            accelerator.save_state(output_dir=str(temp_path))
            if hasattr(accelerator, "wait_for_everyone"):
                accelerator.wait_for_everyone()
            if getattr(accelerator, "is_main_process", True):
                state_files = [
                    path
                    for path in temp_path.rglob("*")
                    if path.is_file()
                    and path.name not in {_SUCCESS_FILE, _METADATA_FILE, _MANIFEST_FILE}
                ]
                if not state_files:
                    raise CheckpointError("Accelerate produced no checkpoint state files")
                manifest = {
                    "schema_version": 1,
                    "files": [
                        {
                            "path": path.relative_to(temp_path).as_posix(),
                            "size": path.stat().st_size,
                            "sha256": _file_sha256(path),
                        }
                        for path in sorted(state_files)
                    ],
                }
                metadata: dict[str, Any] = {
                    "schema_version": 1,
                    "global_step": progress.global_step,
                    "epoch": progress.epoch,
                    "batch_in_epoch": progress.batch_in_epoch,
                    "samples_seen": progress.samples_seen,
                    "best_metric": progress.best_metric,
                    "run_id": progress.run_id,
                    "reason": safe_reason,
                    "created_at": created_at,
                    "config_sha256": config_fingerprint(config),
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                }
                if extra_metadata:
                    metadata["extra"] = dict(extra_metadata)
                (temp_path / _METADATA_FILE).write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (temp_path / _MANIFEST_FILE).write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (temp_path / _SUCCESS_FILE).write_text("complete\n", encoding="utf-8")
                os.replace(temp_path, final_path)
                self._prune()
            if hasattr(accelerator, "wait_for_everyone"):
                accelerator.wait_for_everyone()
        except BaseException:
            LOGGER.exception(
                "Checkpoint save failed; incomplete temp data remains at %s", temp_path
            )
            raise
        LOGGER.info("Saved %s checkpoint at %s", safe_reason, final_path)
        return final_path.resolve()

    def restore(
        self,
        accelerator: Any,
        reference: str | Path | None,
        current_config: Mapping[str, Any] | None = None,
        strict_config_match: bool = True,
    ) -> dict[str, Any] | None:
        checkpoint = self.resolve(reference)
        if checkpoint is None:
            return None
        self._validate_files(checkpoint, verify_hashes=True)
        metadata = json.loads((checkpoint / _METADATA_FILE).read_text(encoding="utf-8"))
        saved_fingerprint = metadata.get("config_sha256")
        if current_config is not None and saved_fingerprint:
            current_fingerprint = config_fingerprint(current_config)
            if current_fingerprint != saved_fingerprint:
                message = (
                    "Checkpoint configuration fingerprint differs from the current config: "
                    f"saved={saved_fingerprint}, current={current_fingerprint}"
                )
                if strict_config_match:
                    raise CheckpointError(message)
                LOGGER.warning(message)
        accelerator.load_state(str(checkpoint))
        LOGGER.info("Restored checkpoint from %s", checkpoint)
        return metadata

    def _validate_files(self, checkpoint: Path, *, verify_hashes: bool) -> None:
        if not (checkpoint / _SUCCESS_FILE).is_file():
            raise CheckpointError(
                f"Checkpoint is incomplete (missing {_SUCCESS_FILE}): {checkpoint}"
            )
        if not (checkpoint / _METADATA_FILE).is_file():
            raise CheckpointError(f"Checkpoint metadata is missing: {checkpoint}")
        manifest_path = checkpoint / _MANIFEST_FILE
        if not manifest_path.is_file():
            raise CheckpointError(f"Checkpoint manifest is missing: {checkpoint}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"Checkpoint manifest is unreadable: {checkpoint}") from exc
        if not isinstance(files, list) or not files:
            raise CheckpointError(f"Checkpoint manifest contains no state files: {checkpoint}")
        for entry in files:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                raise CheckpointError(f"Checkpoint manifest entry is invalid: {checkpoint}")
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise CheckpointError(f"Checkpoint manifest contains an unsafe path: {relative}")
            state_file = checkpoint / relative
            if not state_file.is_file() or state_file.stat().st_size != int(entry.get("size", -1)):
                raise CheckpointError(
                    f"Checkpoint state file is missing or truncated: {state_file}"
                )
            if verify_hashes and _file_sha256(state_file) != entry.get("sha256"):
                raise CheckpointError(f"Checkpoint state file hash mismatch: {state_file}")

    def _prune(self) -> None:
        records = self.list_complete()
        for record in records[: max(0, len(records) - self.keep_last)]:
            # Only records discovered by the strict managed-directory pattern reach here.
            shutil.rmtree(record.path)
            LOGGER.info("Removed rotated checkpoint %s", record.path)


def _file_sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StopController:
    """Turn OS signals into a stop request checked at a safe optimizer boundary."""

    def __init__(self) -> None:
        self.requested = False
        self.reason = "signal"
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if self.requested:
            raise KeyboardInterrupt("Second termination signal received")
        self.requested = True
        try:
            self.reason = signal.Signals(signum).name.lower()
        except ValueError:
            self.reason = f"signal-{signum}"
        LOGGER.warning("%s received; checkpointing at the next safe training boundary", self.reason)

    def __enter__(self) -> StopController:
        for signum in (signal.SIGINT, signal.SIGTERM):
            if hasattr(signal, signal.Signals(signum).name):
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()

    def raise_if_requested(self) -> None:
        if self.requested:
            raise GracefulStopRequested(self.reason)


@contextmanager
def emergency_checkpoint_guard(
    save_callback: Callable[[str], Any],
    *,
    save_on_interrupt: bool = True,
    save_on_exception: bool = True,
) -> Iterator[None]:
    """Attempt an emergency save while preserving and re-raising the original failure."""

    try:
        yield
    except (GracefulStopRequested, KeyboardInterrupt) as exc:
        if save_on_interrupt:
            try:
                reason = str(exc) or "interrupt"
                save_callback(reason)
            except BaseException:
                LOGGER.exception("Emergency checkpoint after interruption also failed")
        raise
    except Exception as exc:
        if save_on_exception:
            try:
                save_callback(f"exception-{type(exc).__name__}")
            except BaseException:
                LOGGER.exception("Emergency checkpoint after exception also failed")
        raise
