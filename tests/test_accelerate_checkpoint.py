from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator

from scenearchitect.training.checkpointing import CheckpointManager, TrainingProgress


def test_accelerate_restores_model_optimizer_scheduler_and_progress(tmp_path: Path) -> None:
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    progress = TrainingProgress(global_step=7, epoch=2, batch_in_epoch=3, samples_seen=11)
    accelerator.register_for_checkpointing(progress)

    batch = torch.ones(2, 2)
    loss = model(batch).sum()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected_weight = accelerator.unwrap_model(model).weight.detach().clone()
    expected_lr = scheduler.get_last_lr()[0]

    manager = CheckpointManager(tmp_path, keep_last=1)
    manager.save(accelerator, progress, {"schema_version": 1, "run": {"name": "tiny"}})
    with torch.no_grad():
        accelerator.unwrap_model(model).weight.zero_()
    progress.global_step = 999
    scheduler.step()

    manager.restore(
        accelerator,
        "latest",
        {"schema_version": 1, "run": {"name": "tiny"}},
    )
    assert torch.equal(accelerator.unwrap_model(model).weight, expected_weight)
    assert progress.global_step == 7
    assert progress.epoch == 2
    assert progress.batch_in_epoch == 3
    assert scheduler.get_last_lr()[0] == expected_lr
