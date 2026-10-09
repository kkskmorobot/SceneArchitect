# SceneArchitect

SceneArchitect 是一个面向单卡 16GB 显存的、可控动漫风自然场景与建筑生成工程。项目采用 SDXL、LoRA 与 ControlNet，目标输入包括文本、草图、Canny、Lineart、Depth 及可选 Segmentation，并提供多控制组合、高分辨率二阶段修复、评测与本地 Gradio UI。

## 启动前端

普通使用：双击 `BuShu/start.bat`。PyCharm 使用：打开 `BuShu/start.py`，确认解释器是项目 `.venv`，然后右键运行。两种方式都会打开 `http://127.0.0.1:7860`。

前端生成的图片、实际控制图和参数 JSON 统一保存在 `BuShu/output/<本次生成 ID>/`，不会混入训练输出。

如果 Depth ControlNet 下载中断，在项目根目录运行 `pwsh -ExecutionPolicy Bypass -File ".\BuShu\download_depth_controlnet.ps1"`；反复运行同一条命令会从 `.part` 断点继续，完成后自动校验 SHA-256。

前端提供五个正式模型版本：

- `v1`：SDXL + 第一版建筑 LoRA
- `v2-人物`：SDXL + 第二版人物 LoRA
- `v2-建筑`：SDXL + 第二版建筑 LoRA
- `v3-人物`：Animagine XL 4.0 + 人物 LoRA
- `v3-建筑`：Animagine XL 4.0 + 建筑 LoRA

所有版本都支持文字生成；上传草稿后可选择 `canny`、`depth` 或 `canny-full` 结构控制。前两项使用 SDXL Small ControlNet，`canny-full` 使用完整尺寸的官方 SDXL Canny ControlNet。切换 SDXL/Animagine 系列时会释放上一条推理管线，适配单卡 16GB 显存。

## 技术路线

- 基座：`stabilityai/stable-diffusion-xl-base-1.0`。先保持官方基座不变，再通过用户自有且授权清晰的数据训练场景/建筑 LoRA。
- 训练：仅训练 UNet LoRA；默认不训练两个文本编码器。推荐 bf16、768 分辨率桶、batch size 1、梯度累积 4、gradient checkpointing、8-bit AdamW（WSL2/Linux）。
- 控制：优先复用预训练 SDXL ControlNet。Canny 与 Depth 给出默认模型；Lineart 和 Segmentation 在确认模型许可与兼容性后由配置启用，不静默下载来源不明权重。
- 推理：单控制可在 1024 运行；双控制默认启用 CPU offload、VAE tiling/slicing。高分辨率采用“基础生成 + img2img/tile refine”，不在一次前向中硬顶超大图。
- 注意力：RTX 5080 默认使用 PyTorch SDPA。xFormers 仅作为经过环境验证后的可选加速项。
- UI：Gradio；配置：YAML；训练状态：Accelerate `save_state/load_state` + SceneArchitect 元数据和完整性标记。

完整设计见 [docs/architecture.md](docs/architecture.md)，16GB 降配策略见 [docs/16gb_vram_guide.md](docs/16gb_vram_guide.md)，断点语义见 [docs/checkpointing.md](docs/checkpointing.md)。

## 环境安装

优先使用 Ubuntu 24.04 或 WSL2，并在 PyCharm 中选择 WSL Python 解释器。RTX 5080 属于 Blackwell 架构，先通过 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 安装带 CUDA 12.8 或更新兼容运行时的 PyTorch；PyTorch 2.7 是官方首次提供 Blackwell 与 CUDA 12.8 wheel 的版本，因此不要使用更老版本。

```bash
cd /mnt/e/App/DaiMa/JetBrains/Project/TuXiangAI
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 示例；若官网当前给出更新命令，应采用官网命令。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train,inference,data,ui,eval,dev]"
scene-architect doctor --config configs/project.yaml
```

项目没有把 PyTorch 写死进 `pyproject.toml`，这是为了防止 pip 在 RTX 5080 机器上误装 CPU wheel 或不支持 Blackwell 的旧 CUDA wheel。

## 当前可运行命令

```bash
# 环境与配置检查
python scripts/doctor.py --config configs/project.yaml

# 展开 YAML、环境变量和命令行覆盖项
scene-architect show-config configs/train/sdxl_lora_16gb.yaml \
  --set training.max_train_steps=2000 \
  --set checkpointing.resume_from=latest

# 查看、解析训练断点
scene-architect checkpoints outputs/training/architecture_style --list
scene-architect checkpoints outputs/training/architecture_style --resolve latest

# 数据准备：默认读取 data/data/data
python scripts/prepare_data.py --config configs/data/preprocess.yaml

# 保留原始 caption，生成场景强化且可审计的 train_scene/val_scene 元数据
python scripts/curate_captions.py --config configs/data/caption_curation.yaml

# 配置、恢复路径和数据就绪检查，不加载 PyTorch/模型
python scripts/train_lora.py --config configs/train/sdxl_lora_16gb.yaml --dry-run

# 正式训练；latest 只选择完整且哈希验证通过的 checkpoint
python scripts/train_lora.py --config configs/train/sdxl_lora_16gb.yaml --resume latest

# 建议先做 20 步 GPU 冒烟，并禁用耗时验证图
python scripts/train_lora.py --config configs/train/sdxl_lora_16gb.yaml \
  --max-train-steps 20 --resume none --set validation.every_n_steps=0
```

## 断点与中断恢复

训练阶段会遵守以下约定：

1. 每 `checkpointing.every_n_steps` 保存一次完整训练状态。
2. 保存内容包含 LoRA 模型、优化器、学习率调度器、AMP scaler、随机数状态、epoch/global step 与配置指纹。
3. Ctrl+C、SIGTERM 或未处理异常会尝试写入一个紧急断点；信号只在安全训练步边界处理。
4. 每个状态文件都记录在 `manifest.json` 中并带大小与 SHA-256；恢复时完整验签。目录只有写入 `_SUCCESS` 后才被视为可恢复，`.tmp` 目录永远不会被 `latest` 选中。
5. `resume_from: latest` 自动读取最新完整断点；也可传入明确路径。配置指纹不匹配默认拒绝恢复，避免把旧实验状态混入新实验。
6. `keep_last` 只轮转完整训练状态；最终导出的 LoRA 权重不受它影响。

训练循环已经通过 `CheckpointManager` 与 `StopController` 接入该机制；`final/` 仅供推理，不会被当作可续训 checkpoint。

## 数据约定

```text
data/
├── raw/images/                  # 用户拥有使用权的原始图像
├── interim/                     # 清洗、caption 和临时控制图
└── processed/
    ├── images/
    ├── controls/{canny,lineart,depth,seg}/
    └── metadata/{train,val}.jsonl
```

每条 JSONL 元数据包含 `image`, `caption`, `split`, `width`, `height`, `bucket_width`, `bucket_height`, `scene_type`, `architecture_style`, `weather`, `time_of_day`, `camera` 和各控制图路径。项目不提供或伪造大规模数据；请确认训练图片、caption、参考图和模型权重的许可范围。

默认 caption 优先读取同名 `.txt`，没有 sidecar 时从文件名生成可人工复核的 caption。Canny 和基础 Lineart 默认本地生成；Depth/Seg 在 `configs/data/preprocess.yaml` 中显式启用后才下载相应模型。

## 分阶段运行入口

| 阶段 | 入口 | 当前状态 |
|---|---|---|
| A-B 设计/骨架 | `scripts/doctor.py`、CLI | 已完成 |
| C 数据预处理 | `scripts/prepare_data.py`、`scripts/curate_captions.py` | 已实现；包含场景强化 caption 策展 |
| D LoRA 训练 | `scripts/train_lora.py` | 已实现；训练读取 `train_scene.jsonl` |
| E 推理 | `scripts/infer.py` | 阶段门禁 |
| F Gradio | `BuShu/launch_ui.py` | 阶段门禁 |
| G 评测 | `scripts/evaluate.py` | 阶段门禁 |

## 推荐实施顺序

1. 把自有且授权清晰的图片放入 `data/raw/images`，可选放同名 `.txt` caption，然后运行阶段 C。
2. 先执行训练 dry-run，再做 20 步 `architecture_style` GPU 冒烟；确认显存和断点恢复后再开始正式长跑。
3. 第二批 LoRA：古风、赛博、雨夜、雪景等 `atmosphere_style`，每个目标保持独立输出目录。
4. 阶段 E：先文本和单 Canny，再加入 Depth 与双 ControlNet；最后加入二阶段 refine。
5. 阶段 F-G：UI、CLIPScore、边缘/深度一致性、人工评分表与 ablation 导出。

## 16GB 常见问题

- 训练 OOM：保持 `train_text_encoders: false`，先把 bucket 上限降到 704/768，再提高梯度累积补偿有效 batch；必要时启用 8-bit AdamW 和 latent cache。
- 推理 OOM：一次只加载一个 ControlNet；双控制时启用 `model_cpu_offload`，先生成 768，再分块 refine 到 1024/1536。
- xFormers 安装失败：保持 `attention_backend: sdpa`，PyTorch 2.x 已自带 scaled dot product attention。
- Windows 原生 bitsandbytes/编译依赖不稳定：使用 WSL2；项目路径可直接通过 `/mnt/e/...` 访问。
- 模型下载失败：先登录 Hugging Face，确认模型许可，再设置 `HF_TOKEN`；不要把 token 写进 YAML 或 Git。

## 项目结构

```text
configs/      训练、推理、评测、预设与 Accelerate 配置
data/         原始、中间、处理后数据（默认不入 Git）
docs/         架构、显存、断点与开发阶段文档
models/       本地权重与 Hugging Face 缓存（默认不入 Git）
outputs/      训练、生成、评测结果（默认不入 Git）
BuShu/        前端部署入口、专用配置、启动说明与 output/
scripts/      用户直接运行的薄入口
src/          可测试、可复用的 Python 包
tests/        不下载大模型的快速测试
```

## 后续扩展

- 参考图风格：IP-Adapter SDXL，但放在核心路径稳定之后。
- 局部重绘：SDXL inpainting + mask 编辑。
- 更强结构控制：经许可核对的 Lineart/Seg ControlNet 或 ControlNet Union。
- 研究：不同控制组合/强度的结构一致性 ablation、LoRA 目标拆分与融合、自动候选排序。

## 许可提醒

本仓库代码计划使用 Apache-2.0；模型和数据不自动继承代码许可。SDXL 1.0 使用 CreativeML Open RAIL++-M，ControlNet、LoRA、caption 模型和数据集各有独立许可。用于比赛、公开演示或商业用途前必须逐项核对。
