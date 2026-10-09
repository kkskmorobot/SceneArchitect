"""Gradio UI for the five deployed SceneArchitect model profiles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from scenearchitect.inference.generator import (
    GenerationError,
    SceneArchitectGenerator,
    inspect_assets,
)


def build_app(config: dict[str, Any], repository_root: str | Path) -> Any:
    import gradio as gr

    root = Path(repository_root).resolve()
    holder: dict[str, SceneArchitectGenerator] = {}
    presets = {str(key): str(value) for key, value in config.get("presets", {}).items()}
    profiles = {
        str(key): dict(value)
        for key, value in config.get("profiles", {}).items()
        if isinstance(value, Mapping)
    }
    if not profiles:
        profiles = {
            str(key): {
                "label": str(value.get("label", key)),
                "model": str(key),
                "adapter": None,
                "default_weight": 0.0,
            }
            for key, value in config.get("models", {}).items()
            if isinstance(value, Mapping)
        }
    if not profiles:
        raise ValueError("At least one model profile must be configured")

    def generate(
        prompt: str,
        negative_prompt: str,
        profile_key: str,
        style_weight: float,
        sketch: Image.Image | None,
        control_type: str,
        control_scale: float,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        num_images: int,
    ) -> tuple[list[str], str, Image.Image | None, str]:
        if profile_key not in profiles:
            raise gr.Error(f"未知模型版本：{profile_key}")
        profile = profiles[profile_key]
        model_key = str(profile["model"])
        adapter_name = str(profile.get("adapter") or "").strip()
        weights = {adapter_name: float(style_weight)} if adapter_name else {}
        try:
            generator = holder.setdefault(
                "generator", SceneArchitectGenerator(config, repository_root=root)
            )
            result = generator.generate(
                model_key=model_key,
                profile_key=profile_key,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                adapter_weights=weights,
                control_image=sketch,
                control_type=control_type if sketch is not None else None,
                control_scale=control_scale,
                width=int(width),
                height=int(height),
                steps=int(steps),
                guidance_scale=float(guidance),
                seed=int(seed),
                num_images=int(num_images),
            )
        except (GenerationError, OSError, RuntimeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        pretty = json.dumps(result.metadata, ensure_ascii=False, indent=2)
        profile_label = str(profile.get("label", profile_key))
        status = f"{profile_label} 生成完成，参数已保存到 `{result.metadata_path}`"
        return result.image_paths, pretty, result.control_preview, status

    def apply_preset(name: str) -> str:
        return presets.get(name, "")

    def apply_profile_defaults(profile_key: str) -> tuple[int, float, float]:
        profile = profiles[profile_key]
        model = config["models"][str(profile["model"])]
        return (
            int(profile.get("default_steps", model.get("default_steps", 30))),
            float(profile.get("default_guidance", model.get("default_guidance", 6.5))),
            float(profile.get("default_weight", 0.0)),
        )

    ui_config = config.get("ui", {})
    with gr.Blocks(title=str(ui_config.get("title", "SceneArchitect"))) as demo:
        gr.Markdown("# SceneArchitect\n用文字描述或草稿生成动漫人物与建筑场景。")
        with gr.Row():
            with gr.Column(scale=3):
                preset = gr.Dropdown(
                    choices=list(presets), value=next(iter(presets), None), label="场景预设"
                )
                prompt = gr.Textbox(
                    label="图片描述（中文或英文）",
                    lines=4,
                    placeholder="例如：薄雾中的森林神社，远处有一个撑伞的人，电影感广角构图",
                )
                negative = gr.Textbox(label="负面提示词（可留空使用默认值）", lines=2)
                profile_choices = [
                    (str(item.get("label", key)), str(key)) for key, item in profiles.items()
                ]
                initial_profile = profile_choices[0][1]
                model_profile = gr.Dropdown(
                    choices=profile_choices,
                    value=initial_profile,
                    label="模型版本",
                )
                style_weight = gr.Slider(
                    0.0,
                    1.5,
                    value=float(profiles[initial_profile].get("default_weight", 0.85)),
                    step=0.05,
                    label="模型风格强度",
                )
                sketch = gr.Image(
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    label="草稿/边缘图（可选）",
                )
                with gr.Row():
                    control_type = gr.Dropdown(
                        choices=list(inspect_assets(config, root).controlnets),
                        value="canny",
                        label="控制类型（canny/depth 为 Small，canny-full 为完整模型）",
                    )
                    control_scale = gr.Slider(0.0, 1.5, value=0.75, step=0.05, label="草稿约束强度")
            with gr.Column(scale=2):
                with gr.Row():
                    width = gr.Dropdown([512, 640, 768, 896, 1024], value=768, label="宽度")
                    height = gr.Dropdown([512, 640, 768, 896, 1024], value=768, label="高度")
                steps = gr.Slider(10, 60, value=30, step=1, label="采样步数")
                guidance = gr.Slider(1.0, 12.0, value=6.5, step=0.1, label="提示词强度")
                seed = gr.Number(value=-1, precision=0, label="Seed（-1 随机）")
                num_images = gr.Slider(1, 4, value=1, step=1, label="生成数量")
                generate_button = gr.Button("生成图片", variant="primary")
                status = gr.Markdown("等待生成")
                metadata = gr.Code(label="生成参数 JSON", language="json")
        with gr.Row():
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="生成结果",
                    columns=1,
                    height="auto",
                    type="filepath",
                )
            with gr.Column(scale=1):
                control_preview = gr.Image(
                    label="实际送入 ControlNet 的控制图",
                    type="pil",
                    height="auto",
                )

        preset.change(apply_preset, inputs=preset, outputs=prompt)
        model_profile.change(
            apply_profile_defaults,
            inputs=model_profile,
            outputs=[steps, guidance, style_weight],
        )
        generate_button.click(
            generate,
            inputs=[
                prompt,
                negative,
                model_profile,
                style_weight,
                sketch,
                control_type,
                control_scale,
                width,
                height,
                steps,
                guidance,
                seed,
                num_images,
            ],
            outputs=[gallery, metadata, control_preview, status],
        )
    demo.queue(default_concurrency_limit=1, max_size=int(ui_config.get("max_queue", 8)))
    return demo
