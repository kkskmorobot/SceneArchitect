from __future__ import annotations

import json
from pathlib import Path

import pytest

from scenearchitect.training.checkpointing import (
    CheckpointError,
    CheckpointManager,
    TrainingProgress,
)


class FakeAccelerator:
    is_main_process = True

    def __init__(self) -> None:
        self.loaded: Path | None = None

    def wait_for_everyone(self) -> None:
        return None

    def save_state(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "accelerator_state.bin").write_bytes(b"test-state")

    def load_state(self, input_dir: str) -> None:
        self.loaded = Path(input_dir)


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"name": "test", "output_dir": "ignored"},
        "checkpointing": {"resume_from": "latest", "auto_resume_if_available": True},
    }


def test_atomic_save_latest_restore_and_rotation(tmp_path: Path) -> None:
    accelerator = FakeAccelerator()
    manager = CheckpointManager(tmp_path, keep_last=2)
    progress = TrainingProgress()

    for step in (10, 20, 30):
        progress.global_step = step
        saved = manager.save(accelerator, progress, _config(), reason="periodic")
        assert (saved / "_SUCCESS").is_file()

    records = manager.list_complete()
    assert [record.global_step for record in records] == [20, 30]
    latest = manager.latest()
    assert latest is not None and latest.name == "checkpoint-step-000000000030"

    metadata = manager.restore(accelerator, "latest", _config())
    assert metadata is not None and metadata["global_step"] == 30
    assert accelerator.loaded == latest


def test_incomplete_checkpoint_is_ignored(tmp_path: Path) -> None:
    incomplete = tmp_path / "checkpoint-step-000000000999"
    incomplete.mkdir()
    (incomplete / "metadata.json").write_text(json.dumps({"global_step": 999}), encoding="utf-8")
    assert CheckpointManager(tmp_path).latest() is None


def test_tampered_state_is_rejected_on_restore(tmp_path: Path) -> None:
    accelerator = FakeAccelerator()
    manager = CheckpointManager(tmp_path)
    saved = manager.save(accelerator, TrainingProgress(global_step=5), _config())
    (saved / "accelerator_state.bin").write_bytes(b"tampered")

    with pytest.raises(CheckpointError, match="missing or truncated|hash mismatch"):
        manager.restore(accelerator, saved, _config())
