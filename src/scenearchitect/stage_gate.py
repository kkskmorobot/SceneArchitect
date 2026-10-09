"""Current implementation status for the staged SceneArchitect delivery."""

from __future__ import annotations

STAGE_STATUS = {
    "A": "complete: architecture and technical route",
    "B": "complete: project skeleton, config, diagnostics, checkpoint infrastructure",
    "C": "complete: validation, captions, controls, buckets, and metadata",
    "D": "complete: SDXL UNet LoRA training with full-state resume",
    "E": "complete: text generation, Canny/depth ControlNet, dual LoRA loading, metadata",
    "F": "complete: local Gradio text/sketch generation UI",
    "G": "pending: evaluation and final documentation",
}


class StageNotImplementedError(RuntimeError):
    pass


def require_stage(stage: str, feature: str) -> None:
    status = STAGE_STATUS.get(stage, "unknown")
    if not status.startswith("complete"):
        raise StageNotImplementedError(
            f"{feature} belongs to stage {stage}, which is currently {status}. "
            "The command stopped before loading a model or writing fake outputs."
        )
