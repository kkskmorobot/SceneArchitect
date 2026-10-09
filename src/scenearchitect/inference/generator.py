"""Lazy SDXL, LoRA, and ControlNet inference for a single 16 GB GPU."""

from __future__ import annotations

import gc
import json
import logging
import random
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

LOGGER = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when generation assets or request parameters are invalid."""


@dataclass(frozen=True)
class AssetStatus:
    models: dict[str, str]
    adapters: dict[str, dict[str, str | None]]
    controlnets: dict[str, str]


@dataclass(frozen=True)
class GenerationResult:
    image_paths: list[str]
    metadata_path: str
    metadata: dict[str, Any]
    control_preview: Image.Image | None


def _resolve_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return (repository_root / path).resolve() if not path.is_absolute() else path.resolve()


def _adapter_is_complete(path: Path) -> bool:
    return path.is_dir() and (path / "_SUCCESS").is_file() and any(path.glob("*.safetensors"))


def _model_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = config.get("models")
    if isinstance(models, Mapping) and models:
        return {
            str(key): dict(value)
            for key, value in models.items()
            if isinstance(value, Mapping)
        }
    legacy = dict(config["model"])
    legacy["adapters"] = config.get("adapters", {})
    return {"sdxl": legacy}


def inspect_assets(config: dict[str, Any], repository_root: str | Path) -> AssetStatus:
    """Resolve available adapters without importing Torch or loading a model."""

    root = Path(repository_root).resolve()
    models = _model_configs(config)
    adapters: dict[str, dict[str, str | None]] = {}
    for model_key, model_config in models.items():
        model_adapters: dict[str, str | None] = {}
        for name, adapter_config in model_config.get("adapters", {}).items():
            selected = None
            for candidate in adapter_config.get("candidates", []):
                path = _resolve_path(candidate, root)
                if _adapter_is_complete(path):
                    selected = str(path)
                    break
            model_adapters[str(name)] = selected
        adapters[model_key] = model_adapters
    controlnets: dict[str, str] = {}
    for name, item in config.get("controlnets", {}).items():
        if not isinstance(item, Mapping) or not item.get("enabled") or not item.get("model_id"):
            continue
        reference = str(item["model_id"])
        local_reference = _resolve_path(reference, root)
        controlnets[str(name)] = str(local_reference) if local_reference.is_dir() else reference
    return AssetStatus(
        models={key: str(value["base_model"]) for key, value in models.items()},
        adapters=adapters,
        controlnets=controlnets,
    )


def prepare_control_image(
    image: Image.Image, control_type: str, width: int, height: int
) -> Image.Image:
    """Convert an uploaded sketch/control image into a pipeline-ready RGB control."""

    if image is None:
        raise GenerationError("Control image is required when control is enabled")
    source = ImageOps.fit(
        ImageOps.exif_transpose(image).convert("RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
    )
    if control_type in {"canny", "canny-full"}:
        try:
            import cv2
        except ImportError as exc:
            raise GenerationError("Canny control requires opencv-python-headless") from exc
        gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2), mode="RGB")
    if control_type == "depth":
        return ImageOps.grayscale(source).convert("RGB")
    if control_type == "lineart":
        gray = ImageOps.grayscale(source)
        return ImageOps.invert(gray).convert("RGB")
    raise GenerationError(f"Unsupported control type: {control_type}")


def _validate_dimensions(width: int, height: int) -> None:
    if width % 64 or height % 64:
        raise GenerationError("Width and height must be multiples of 64")
    if min(width, height) < 512 or max(width, height) > 1024:
        raise GenerationError("16 GB preset supports dimensions from 512 through 1024")


class SceneArchitectGenerator:
    """Own one lazy pipeline at a time and serialize access to the GPU."""

    def __init__(self, config: dict[str, Any], repository_root: str | Path) -> None:
        self.config = config
        self.repository_root = Path(repository_root).resolve()
        self.assets = inspect_assets(config, self.repository_root)
        self.models = _model_configs(config)
        self._pipeline: Any | None = None
        self._pipeline_mode: str | None = None
        self._pipeline_model_key: str | None = None
        self._loaded_adapters: set[str] = set()
        self._lock = threading.RLock()

    def _torch_dtype(self, torch: Any) -> Any:
        dtype = str(self.config.get("memory", {}).get("dtype", "bf16"))
        if dtype == "bf16":
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise GenerationError("BF16 inference was requested but CUDA BF16 is unavailable")
            return torch.bfloat16
        if dtype == "fp16":
            return torch.float16
        return torch.float32

    def _release_pipeline(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
        self._pipeline = None
        self._pipeline_mode = None
        self._pipeline_model_key = None
        self._loaded_adapters.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_pipeline(self, model_key: str, control_type: str | None) -> Any:
        if model_key not in self.models:
            raise GenerationError(f"Unknown base model: {model_key}")
        mode = control_type or "text"
        if (
            self._pipeline is not None
            and self._pipeline_mode == mode
            and self._pipeline_model_key == model_key
        ):
            return self._pipeline

        self._release_pipeline()
        try:
            import torch
            from diffusers import (
                ControlNetModel,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                StableDiffusionXLControlNetPipeline,
                StableDiffusionXLPipeline,
            )
        except ImportError as exc:
            raise GenerationError("Inference dependencies are not installed") from exc
        if not torch.cuda.is_available():
            raise GenerationError("SDXL inference requires a CUDA GPU in this 16 GB profile")

        model = self.models[model_key]
        model_reference = str(model["base_model"])
        local_model = _resolve_path(model_reference, self.repository_root)
        if local_model.is_dir():
            model_reference = str(local_model)
        dtype = self._torch_dtype(torch)
        common = {
            "revision": model.get("revision"),
            "torch_dtype": dtype,
            "local_files_only": bool(model.get("local_files_only", False)),
            "use_safetensors": True,
        }
        if model.get("variant"):
            common["variant"] = model["variant"]
        if control_type:
            if control_type not in self.assets.controlnets:
                raise GenerationError(
                    f"ControlNet {control_type!r} is not enabled or has no verified model id"
                )
            control_config = self.config["controlnets"][control_type]
            controlnet_options = {
                "torch_dtype": dtype,
                "use_safetensors": True,
                "local_files_only": common["local_files_only"],
            }
            if control_config.get("variant"):
                controlnet_options["variant"] = control_config["variant"]
            controlnet = ControlNetModel.from_pretrained(
                self.assets.controlnets[control_type], **controlnet_options
            )
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_reference, controlnet=controlnet, **common
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(model_reference, **common)

        if model.get("scheduler") == "euler_a":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config
            )
        else:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
        pipeline.set_progress_bar_config(disable=True)
        memory = self.config.get("memory", {})
        if memory.get("vae_slicing", True):
            pipeline.enable_vae_slicing()
        if memory.get("vae_tiling", True):
            pipeline.enable_vae_tiling()
        if memory.get("model_cpu_offload", True):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cuda")

        self._pipeline = pipeline
        self._pipeline_mode = mode
        self._pipeline_model_key = model_key
        for name, path in self.assets.adapters.get(model_key, {}).items():
            if path:
                pipeline.load_lora_weights(path, adapter_name=name)
                self._loaded_adapters.add(name)
        return pipeline

    def _apply_adapters(self, pipeline: Any, weights: dict[str, float]) -> dict[str, float]:
        active = {name: float(weight) for name, weight in weights.items() if float(weight) > 0.0}
        missing = [name for name in active if name not in self._loaded_adapters]
        if missing:
            raise GenerationError(
                "Requested adapter is not trained/available yet: " + ", ".join(sorted(missing))
            )
        if not active:
            if hasattr(pipeline, "disable_lora"):
                pipeline.disable_lora()
            return {}
        if hasattr(pipeline, "enable_lora"):
            pipeline.enable_lora()
        pipeline.set_adapters(list(active), adapter_weights=list(active.values()))
        return active

    def _prompt_with_prefixes(
        self, model_key: str, prompt: str, weights: dict[str, float]
    ) -> str:
        parts: list[str] = []
        model = self.models[model_key]
        for name, weight in weights.items():
            if weight <= 0:
                continue
            prefix = str(model.get("adapters", {}).get(name, {}).get("prompt_prefix", ""))
            if prefix and prefix.casefold() not in prompt.casefold():
                parts.append(prefix)
        parts.append(prompt.strip())
        suffix = str(model.get("prompt_suffix", "")).strip()
        if suffix:
            existing = {
                tag.strip().casefold()
                for part in parts
                for tag in part.split(",")
                if tag.strip()
            }
            missing_suffix = [
                tag.strip()
                for tag in suffix.split(",")
                if tag.strip() and tag.strip().casefold() not in existing
            ]
            if missing_suffix:
                parts.append(", ".join(missing_suffix))
        return ", ".join(part for part in parts if part)

    def generate(
        self,
        *,
        model_key: str = "sdxl",
        profile_key: str | None = None,
        prompt: str,
        negative_prompt: str | None = None,
        adapter_weights: dict[str, float] | None = None,
        control_image: Image.Image | None = None,
        control_type: str | None = None,
        control_scale: float | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        num_images: int = 1,
    ) -> GenerationResult:
        if not prompt or not prompt.strip():
            raise GenerationError("Prompt cannot be empty")
        if model_key not in self.models:
            raise GenerationError(f"Unknown base model: {model_key}")
        model_defaults = self.models[model_key]
        generation = self.config.get("generation", {})
        width = int(width or generation.get("width", 1024))
        height = int(height or generation.get("height", 1024))
        _validate_dimensions(width, height)
        steps = int(
            steps
            if steps is not None
            else model_defaults.get("default_steps", generation.get("num_inference_steps", 30))
        )
        guidance_scale = float(
            guidance_scale
            if guidance_scale is not None
            else model_defaults.get("default_guidance", generation.get("guidance_scale", 6.5))
        )
        if not 1 <= steps <= 80:
            raise GenerationError("Inference steps must be between 1 and 80")
        if not 1 <= num_images <= 4:
            raise GenerationError("num_images must be between 1 and 4")
        if seed is None or int(seed) < 0:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
        seed = int(seed)
        adapter_weights = adapter_weights or {}

        with self._lock:
            import torch

            pipeline = self._load_pipeline(
                model_key, control_type if control_image is not None else None
            )
            active_adapters = self._apply_adapters(pipeline, adapter_weights)
            effective_prompt = self._prompt_with_prefixes(model_key, prompt, active_adapters)
            negative = negative_prompt or str(generation.get("negative_prompt", ""))
            generators = [
                torch.Generator(device="cuda").manual_seed(seed + i) for i in range(num_images)
            ]
            prepared_control = None
            kwargs: dict[str, Any] = {}
            if control_image is not None:
                if not control_type:
                    raise GenerationError("Select a control type for the uploaded sketch")
                prepared_control = prepare_control_image(control_image, control_type, width, height)
                default_scale = self.config["controlnets"][control_type].get("default_scale", 0.75)
                kwargs["image"] = prepared_control
                kwargs["controlnet_conditioning_scale"] = float(
                    control_scale if control_scale is not None else default_scale
                )
            result = pipeline(
                prompt=effective_prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generators,
                **kwargs,
            )
            images = list(result.images)

        output = self.config.get("output", {})
        output_root = _resolve_path(
            output.get("directory", "BuShu/output"), self.repository_root
        )
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "profile_key": profile_key,
            "model_key": model_key,
            "base_model": self.assets.models[model_key],
            "prompt": prompt,
            "effective_prompt": effective_prompt,
            "negative_prompt": negative,
            "adapter_weights": active_adapters,
            "adapter_paths": {
                name: self.assets.adapters[model_key][name] for name in active_adapters
            },
            "control_type": control_type if prepared_control is not None else None,
            "control_scale": kwargs.get("controlnet_conditioning_scale"),
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seeds": [seed + index for index in range(num_images)],
        }
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("SceneArchitect", json.dumps(metadata, ensure_ascii=False))
        image_paths: list[str] = []
        for index, image in enumerate(images):
            destination = run_dir / f"image-{index:02d}-seed-{seed + index}.png"
            image.save(destination, pnginfo=png_info)
            image_paths.append(str(destination))
        if prepared_control is not None:
            prepared_control.save(run_dir / "control.png")
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return GenerationResult(
            image_paths=image_paths,
            metadata_path=str(metadata_path),
            metadata=metadata,
            control_preview=prepared_control,
        )

    def close(self) -> None:
        with self._lock:
            self._release_pipeline()
