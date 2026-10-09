# 分阶段开发计划

## 阶段 A：总体设计（已完成）

- 固定 SDXL + LoRA + ControlNet 技术边界。
- 固定 16GB 默认策略、目录契约、实验拆分与断点语义。

验收：设计不依赖未确认的社区权重；每个高显存环节都有降配路径。

## 阶段 B：工程骨架（已完成）

- `pyproject.toml`、YAML 模板、包结构、CLI、doctor、阶段入口。
- 配置覆盖与校验。
- 可独立测试的 CheckpointManager、TrainingProgress 和 StopController。

验收：不安装 PyTorch 也能解析配置和跑基础测试；有 GPU 环境时 doctor 能报告 Blackwell/CUDA 状态。

## 阶段 C：数据预处理（代码已完成，等待用户数据验收）

- 原图格式、尺寸、重复、损坏和透明通道校验。
- caption 自动生成与人工复核工作流；标签词表规范化。
- Canny、基础 Lineart、Depth 控制图；Seg 作为可选插件。
- aspect ratio bucket、固定 seed 的 train/val 划分、JSONL 元数据。

验收：放入自有图片后，一条命令得到可训练目录；重复运行结果稳定；坏图有清单而不是静默丢弃。

## 阶段 D：LoRA 训练（代码已完成，等待数据后 GPU 冒烟）

- 基于 Diffusers SDXL text-to-image LoRA 训练逻辑，UNet adapter 默认 rank 16。
- Accelerate 单卡、bf16、gradient checkpointing、原生 AdamW、梯度累积；bitsandbytes 可选。
- TensorBoard/JSONL 日志、固定验证 prompts、LoRA safetensors 导出。
- 接入现有断点管理器：周期、Ctrl+C、SIGTERM、异常保存与 latest/路径恢复。

验收：小数据 20 步 smoke test；中断后 global step、LR 与优化器状态连续；恢复前后固定 seed 输出流程可重现。

## 阶段 E：推理

- 文本、单 Canny、单 Depth、双 Canny+Depth。
- LoRA 动态加载/卸载、preset、批量与 seed。
- PNG + JSON sidecar；768/1024 基础生成；可选 img2img/tile refine。

验收：每个模式一条 CLI 命令；同配置/seed 可复现；16GB 默认不 OOM。

## 阶段 F：Gradio

- 控制图、控制类型、多控制开关、LoRA/preset 与参数联动。
- 超出安全组合时给出提示；结果、参数与历史可保存。

验收：本地浏览器完成文本、单控制、双控制生成；错误在 UI 中可读。

## 阶段 G：评测与文档

- CLIPScore、Canny 结构一致性、Depth 一致性、人工评分 CSV 模板。
- prompt-only / canny / depth / canny+depth ablation。
- 完整 README、FAQ、许可证清单、模型卡/数据卡模板与展示材料。

验收：一条命令导出 per-sample 和 aggregate CSV/JSON；展示图能追溯到配置、seed 和模型 revision。
