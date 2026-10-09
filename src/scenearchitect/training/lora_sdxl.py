"""Single-GPU, 16GB-oriented SDXL UNet LoRA training implementation."""

from __future__ import annotations

import gc
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenearchitect.config import config_fingerprint
from scenearchitect.data.metadata import load_metadata
from scenearchitect.data.training_dataset import (
    AspectBucketBatchSampler,
    SDXLMetadataDataset,
    collate_sdxl,
)
from scenearchitect.training.checkpointing import (
    CheckpointManager,
    StopController,
    TrainingProgress,
    emergency_checkpoint_guard,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    global_step: int
    samples_seen: int
    resumed_from: str | None
    final_adapter: str
    output_dir: str


def _weight_dtype(torch: Any, precision: str) -> Any:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_optimizer(torch: Any, parameters: list[Any], config: dict[str, Any]) -> Any:
    name = str(config.get("optimizer", "adamw")).lower()
    optimizer_class: Any = torch.optim.AdamW
    if name == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            optimizer_class = bnb.optim.AdamW8bit
        except (ImportError, OSError) as exc:
            LOGGER.warning(
                "bitsandbytes is unavailable; falling back to torch.optim.AdamW: %s", exc
            )
    elif name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    return optimizer_class(
        parameters,
        lr=float(config["learning_rate"]),
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.999))),
        weight_decay=float(config.get("adam_weight_decay", 0.01)),
        eps=float(config.get("adam_epsilon", 1e-8)),
    )


def _encode_prompt(
    text_encoders: list[Any],
    input_ids: list[Any],
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    prompt_embeds_list: list[Any] = []
    pooled_prompt_embeds: Any = None
    for token_ids, text_encoder in zip(input_ids, text_encoders, strict=True):
        output = text_encoder(token_ids, output_hidden_states=True, return_dict=False)
        pooled_prompt_embeds = output[0]
        prompt_embeds_list.append(output[-1][-2])
    prompt_embeds = __import__("torch").cat(prompt_embeds_list, dim=-1)
    return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)


def _save_final_adapter(
    accelerator: Any,
    unet: Any,
    output_dir: Path,
    config: dict[str, Any],
    progress: TrainingProgress,
) -> Path:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    final_dir = output_dir / "final"
    if accelerator.is_main_process:
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(unet)
        state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped))
        StableDiffusionXLPipeline.save_lora_weights(str(final_dir), unet_lora_layers=state)
        metadata = {
            "schema_version": 1,
            "global_step": progress.global_step,
            "samples_seen": progress.samples_seen,
            "config_sha256": config_fingerprint(config),
            "note": "Exported LoRA adapter only; use checkpoint-step-* to resume training.",
        }
        (final_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (final_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    accelerator.wait_for_everyone()
    return final_dir.resolve()


def _run_validation(
    *,
    accelerator: Any,
    unet: Any,
    vae: Any,
    text_encoder_one: Any,
    text_encoder_two: Any,
    tokenizer_one: Any,
    tokenizer_two: Any,
    config: dict[str, Any],
    progress: TrainingProgress,
    weight_dtype: Any,
) -> None:
    if not accelerator.is_main_process:
        return
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    validation = config.get("validation", {})
    prompts = list(validation.get("prompts", []))
    if not prompts:
        return
    output_dir = (
        Path(config["run"]["output_dir"]) / "validation" / f"step-{progress.global_step:012d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        revision=config["model"].get("revision"),
        variant=config["model"].get("variant"),
        torch_dtype=weight_dtype,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(
        int(config["run"].get("seed", 42)) + progress.global_step
    )
    try:
        for prompt_index, prompt in enumerate(prompts):
            for image_index in range(int(validation.get("num_images_per_prompt", 1))):
                with torch.inference_mode():
                    image = pipeline(
                        prompt=prompt,
                        num_inference_steps=int(validation.get("num_inference_steps", 30)),
                        guidance_scale=float(validation.get("guidance_scale", 6.5)),
                        height=int(config["data"]["resolution"]),
                        width=int(config["data"]["resolution"]),
                        generator=generator,
                    ).images[0]
                image.save(output_dir / f"prompt-{prompt_index:02d}-{image_index:02d}.png")
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        unet.train()


def run_training(config: dict[str, Any], repository_root: str | Path) -> TrainingResult:
    """Train an SDXL UNet LoRA and maintain full interrupt-resumable state."""

    import torch
    import torch.nn.functional as functional
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from diffusers.loaders import LoraLoaderMixin
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import compute_snr
    from diffusers.utils import convert_state_dict_to_diffusers
    from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    root = Path(repository_root).resolve()
    output_dir = Path(config["run"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    config["run"]["output_dir"] = str(output_dir)
    logging_dir = output_dir / "logs"
    training = config["training"]
    checkpointing = config["checkpointing"]
    precision = str(training["mixed_precision"])
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        mixed_precision=None if precision == "no" else precision,
        log_with=config.get("logging", {}).get("report_to", "tensorboard"),
        project_config=ProjectConfiguration(
            project_dir=str(output_dir), logging_dir=str(logging_dir)
        ),
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("The current Stage-D implementation supports one process / one GPU")
    if accelerator.device.type != "cuda":
        raise RuntimeError("Formal SDXL training requires CUDA; run --dry-run for CPU preflight")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "training.mixed_precision=bf16 but this CUDA device does not support bf16"
        )

    seed = int(config["run"].get("seed", 42))
    set_seed(seed, device_specific=True)
    if bool(training.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
    model_name = config["model"]["pretrained_model_name_or_path"]
    revision = config["model"].get("revision")
    variant = config["model"].get("variant")
    weight_dtype = _weight_dtype(torch, precision)
    records = load_metadata(config["data"]["train_metadata"], root)

    LOGGER.info("Loading SDXL components from %s", model_name)
    tokenizer_one = AutoTokenizer.from_pretrained(
        model_name, subfolder="tokenizer", revision=revision, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_name, subfolder="tokenizer_2", revision=revision, use_fast=False
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_name, subfolder="scheduler", revision=revision
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
        torch_dtype=weight_dtype,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        model_name,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
        revision=revision,
        variant=variant,
        torch_dtype=torch.float32,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder="unet",
        revision=revision,
        variant=variant,
        torch_dtype=weight_dtype,
    )
    for module in (vae, text_encoder_one, text_encoder_two, unet):
        module.requires_grad_(False)
    lora = config["lora"]
    unet.add_adapter(
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora.get("alpha", lora["rank"])),
            lora_dropout=float(lora.get("dropout", 0.0)),
            init_lora_weights="gaussian",
            target_modules=list(lora.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])),
        )
    )
    if bool(training.get("gradient_checkpointing", True)):
        unet.enable_gradient_checkpointing()
    # The components were instantiated with their final dtypes.  Moving only the
    # device here preserves Diffusers' module-specific dtype policy and avoids a
    # second blanket cast through ModelMixin.to().
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder_one.eval()
    text_encoder_two.eval()

    dataset = SDXLMetadataDataset(
        records,
        tokenizer_one,
        tokenizer_two,
        seed=seed,
        center_crop=bool(config["data"].get("center_crop", False)),
        random_flip=bool(config["data"].get("random_flip", True)),
    )
    sampler = AspectBucketBatchSampler(
        dataset,
        int(training["train_batch_size"]),
        seed=seed,
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_sdxl,
        num_workers=int(config["data"].get("dataloader_num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(config["data"].get("dataloader_num_workers", 4)) > 0,
    )
    trainable_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable LoRA parameters were created")
    optimizer = _build_optimizer(torch, trainable_parameters, training)
    max_train_steps = int(training["max_train_steps"])
    scheduler = get_scheduler(
        str(training.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(training.get("lr_warmup_steps", 0)),
        num_training_steps=max_train_steps,
    )

    def save_model_hook(models: list[Any], weights: list[Any], destination: str) -> None:
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(unet)
            state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped))
            StableDiffusionXLPipeline.save_lora_weights(destination, unet_lora_layers=state)
        while weights:
            weights.pop()

    def load_model_hook(models: list[Any], source: str) -> None:
        while models:
            model = models.pop()
            state, _ = LoraLoaderMixin.lora_state_dict(source)
            unet_state = {
                key.removeprefix("unet."): value
                for key, value in state.items()
                if key.startswith("unet.")
            }
            peft_state = convert_unet_state_dict_to_peft(unet_state)
            incompatible = set_peft_model_state_dict(model, peft_state, adapter_name="default")
            unexpected = getattr(incompatible, "unexpected_keys", None)
            if unexpected:
                raise RuntimeError(f"Unexpected LoRA keys while restoring: {unexpected}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    unet, optimizer, dataloader, scheduler = accelerator.prepare(
        unet, optimizer, dataloader, scheduler
    )
    progress = TrainingProgress()
    accelerator.register_for_checkpointing(progress)
    manager = CheckpointManager(output_dir, keep_last=int(checkpointing["keep_last"]))
    resume_reference = checkpointing.get("resume_from")
    restored = manager.restore(
        accelerator,
        resume_reference,
        current_config=config,
        strict_config_match=bool(checkpointing.get("strict_config_match", True)),
    )
    resumed_from = str(manager.resolve(resume_reference)) if restored else None
    accelerator.init_trackers(
        str(config["run"]["name"]), config={"config_sha256": config_fingerprint(config)}
    )
    LOGGER.info(
        "Training %s LoRA parameters from global_step=%d",
        sum(parameter.numel() for parameter in trainable_parameters),
        progress.global_step,
    )

    last_saved_step = -1

    def save_checkpoint(reason: str) -> Path:
        nonlocal last_saved_step
        saved = manager.save(
            accelerator,
            progress,
            config,
            reason=reason,
            extra_metadata={"resumed_from": resumed_from},
        )
        last_saved_step = progress.global_step
        return saved

    updates_per_epoch = max(1, math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps))
    total_epochs = math.ceil(max_train_steps / updates_per_epoch)
    initial_epoch = progress.epoch
    initial_batch = progress.batch_in_epoch
    stop_controller = StopController()
    pending_samples = 0
    try:
        with (
            stop_controller,
            emergency_checkpoint_guard(
                save_checkpoint,
                save_on_interrupt=bool(checkpointing.get("save_on_interrupt", True)),
                save_on_exception=bool(checkpointing.get("save_on_exception", True)),
            ),
        ):
            for epoch in range(initial_epoch, total_epochs):
                sampler.set_epoch(epoch)
                unet.train()
                for batch_index, batch in enumerate(dataloader):
                    if epoch == initial_epoch and batch_index < initial_batch:
                        continue
                    progress.epoch = epoch
                    pending_samples += int(batch["pixel_values"].shape[0])
                    with accelerator.accumulate(unet):
                        with torch.no_grad():
                            latents = vae.encode(
                                batch["pixel_values"].to(
                                    accelerator.device, dtype=torch.float32, non_blocking=True
                                )
                            ).latent_dist.sample()
                            latents = latents.to(dtype=weight_dtype) * vae.config.scaling_factor
                            prompt_embeds, pooled_prompt_embeds = _encode_prompt(
                                [text_encoder_one, text_encoder_two],
                                [
                                    batch["input_ids_one"].to(accelerator.device),
                                    batch["input_ids_two"].to(accelerator.device),
                                ],
                                dtype=weight_dtype,
                            )
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(
                            0,
                            noise_scheduler.config.num_train_timesteps,
                            (latents.shape[0],),
                            device=latents.device,
                            dtype=torch.long,
                        )
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        time_ids = torch.cat(
                            [
                                batch["original_sizes"],
                                batch["crop_top_lefts"],
                                batch["target_sizes"],
                            ],
                            dim=1,
                        ).to(accelerator.device, dtype=weight_dtype)
                        model_prediction = unet(
                            noisy_latents,
                            timesteps,
                            prompt_embeds,
                            added_cond_kwargs={
                                "time_ids": time_ids,
                                "text_embeds": pooled_prompt_embeds,
                            },
                            return_dict=False,
                        )[0]
                        prediction_type = noise_scheduler.config.prediction_type
                        if prediction_type == "epsilon":
                            target = noise
                        elif prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
                        snr_gamma = training.get("snr_gamma")
                        if snr_gamma is None:
                            loss = functional.mse_loss(
                                model_prediction.float(), target.float(), reduction="mean"
                            )
                        else:
                            snr = compute_snr(noise_scheduler, timesteps)
                            weights = (
                                torch.minimum(snr, float(snr_gamma) * torch.ones_like(snr)) / snr
                            )
                            loss = functional.mse_loss(
                                model_prediction.float(), target.float(), reduction="none"
                            )
                            loss = loss.mean(dim=tuple(range(1, loss.ndim)))
                            loss = (loss * weights).mean()
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                trainable_parameters, float(training.get("max_grad_norm", 1.0))
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        progress.global_step += 1
                        progress.samples_seen += pending_samples
                        pending_samples = 0
                        progress.batch_in_epoch = batch_index + 1
                        if (
                            progress.global_step
                            % int(config.get("logging", {}).get("log_every_n_steps", 10))
                            == 0
                        ):
                            accelerator.log(
                                {
                                    "train/loss": float(loss.detach()),
                                    "train/lr": scheduler.get_last_lr()[0],
                                },
                                step=progress.global_step,
                            )
                        if manager.should_save(
                            progress.global_step, int(checkpointing["every_n_steps"])
                        ):
                            save_checkpoint("periodic")
                        validation_every = int(config.get("validation", {}).get("every_n_steps", 0))
                        if validation_every > 0 and progress.global_step % validation_every == 0:
                            _run_validation(
                                accelerator=accelerator,
                                unet=unet,
                                vae=vae,
                                text_encoder_one=text_encoder_one,
                                text_encoder_two=text_encoder_two,
                                tokenizer_one=tokenizer_one,
                                tokenizer_two=tokenizer_two,
                                config=config,
                                progress=progress,
                                weight_dtype=weight_dtype,
                            )
                        stop_controller.raise_if_requested()
                        if progress.global_step >= max_train_steps:
                            break
                progress.epoch = epoch + 1
                progress.batch_in_epoch = 0
                initial_batch = 0
                if progress.global_step >= max_train_steps:
                    break
    finally:
        accelerator.wait_for_everyone()

    if progress.global_step != last_saved_step:
        save_checkpoint("completed")
    final_adapter = _save_final_adapter(accelerator, unet, output_dir, config, progress)
    accelerator.end_training()
    return TrainingResult(
        global_step=progress.global_step,
        samples_seen=progress.samples_seen,
        resumed_from=resumed_from,
        final_adapter=str(final_adapter),
        output_dir=str(output_dir),
    )
