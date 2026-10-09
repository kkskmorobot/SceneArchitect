# SceneArchitect 总体架构（阶段 A）

## 1. 最终技术路线

SceneArchitect 采用“冻结 SDXL 基座 + 目标拆分 LoRA + 预训练 ControlNet + 两阶段高分辨率”的模块化方案。

生成链路为：

```text
用户输入
  ├─ prompt / negative prompt / preset
  ├─ 草图或原图 ──> Canny / Lineart / Depth / Seg 预处理
  └─ 可选 LoRA
          ↓
SDXL base + 0~2 个 SDXL ControlNet + 1~N 个 LoRA
          ↓
768/1024 基础图
          ↓ 可选
img2img 或 tile refine
          ↓
PNG + JSON 参数 + 历史记录 + 评测输入
```

训练只更新 UNet LoRA adapter；两个 SDXL 文本编码器和 VAE 默认冻结。ControlNet 在第一版只做预训练权重推理，不在 16GB 单卡上训练。LoRA 按职责拆为：

- `architecture_style`：古风、现代、日式、未来等建筑外观与构图语言；
- `landscape_style`：森林、雪山、海岸、庭院等场景视觉语言；
- `atmosphere_style`：天气、光照、昼夜、雨雪与色彩氛围。

独立 LoRA 比一次混训所有概念更容易控制数据分布、权重强度和实验结论，也方便按场景组合加载。

## 2. 为什么适合 16GB RTX 5080

1. LoRA 只训练少量 adapter 参数，基座保持冻结；避免 SDXL 全量微调的显存与存储成本。
2. 默认 bf16 适合新一代 NVIDIA GPU，梯度 checkpointing 以计算换激活显存。
3. 训练从 768 分桶、batch 1、累积 4 起步；先验证可行，再小范围升到 1024，而不是默认把峰值推到极限。
4. 默认不训练文本编码器；Diffusers 官方文档指出文本编码器训练显著增加显存需求。
5. 推理使用 PyTorch SDPA；多 ControlNet 时启用模型 CPU offload、VAE slicing/tiling，并限制同时加载数量。
6. 高分辨率采用二阶段方法，把“结构确定”和“细节增强”拆开，避免一次生成 1536/2048 引发 OOM。
7. 完整断点让长时间单机训练能从优化器、调度器和 RNG 状态继续，而不是只重新加载 LoRA 权重造成学习率或随机序列跳变。

## 3. 完整目录树

```text
TuXiangAI/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── configs/
│   ├── project.yaml
│   ├── accelerate/single_gpu.yaml
│   ├── train/sdxl_lora_16gb.yaml
│   ├── inference/sdxl_controlnet_16gb.yaml
│   ├── evaluation/base.yaml
│   └── presets/scenes.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── architecture.md
│   ├── 16gb_vram_guide.md
│   ├── checkpointing.md
│   └── development_plan.md
├── models/
├── outputs/
├── BuShu/
│   ├── start.py
│   ├── start.bat
│   ├── app.py
│   ├── launch_ui.py
│   ├── frontend_config.yaml
│   ├── download_depth_controlnet.ps1
│   ├── download_full_canny_controlnet.ps1
│   ├── 启动说明.md
│   └── output/
├── logs/
├── scripts/
│   ├── doctor.py
│   ├── prepare_data.py
│   ├── train_lora.py
│   ├── infer.py
│   └── evaluate.py
├── src/scenearchitect/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── logging_utils.py
│   ├── runtime.py
│   ├── stage_gate.py
│   ├── data/
│   ├── training/
│   │   └── checkpointing.py
│   ├── inference/
│   └── evaluation/
└── tests/
    ├── test_config.py
    └── test_checkpointing.py
```

## 4. 关键模块职责

| 模块 | 职责 |
|---|---|
| `config.py` | YAML 读取、环境变量展开、命令行点路径覆盖、基础校验与稳定配置指纹 |
| `runtime.py` | Python/PyTorch/CUDA/显卡/可选依赖诊断，不触发大模型下载 |
| `data` | 数据校验、caption、标签规范化、控制图、分桶、可复现划分 |
| `training` | SDXL LoRA 构建、数据加载、损失、验证、日志、完整断点与恢复 |
| `checkpointing.py` | 原子保存、完整性标记、latest 发现、轮转、配置匹配和安全中断协议 |
| `inference` | 文本、单/多 ControlNet、LoRA 切换、seed、批量、refine 和元数据 |
| `evaluation` | CLIPScore、边缘/深度一致性、人工评分表、CSV/JSON 汇总 |
| `ui` | Gradio 参数联动、显存安全限制、历史与结果保存 |
| `scripts` | 薄入口，只解析参数并调用 `src` 包，避免业务逻辑散落 |

## 5. 关键工程决策

- 基座默认锁定 SDXL 1.0 官方模型 ID，动漫风由自有数据 LoRA 学习；不把许可证和来源未核对的二次元 checkpoint 设为强制默认。
- Canny/Depth 先做，Lineart/Seg 模型 ID 保持显式空值，直到逐项验证 SDXL 兼容性、输入规范和许可。
- 训练和推理配置分离，避免为了推理低显存而意外改变正式训练配置。
- 每个 LoRA 目标有独立输出目录；恢复训练时强校验配置指纹，防止跨实验误续训。
- val 数据只用于训练期间选择；评测模块另建输入清单，避免把演示图和验证图混在一起。

## 6. 依赖分组

- 核心：Python 3.10/3.11、PyYAML、packaging。
- CUDA：PyTorch 2.7+ 的 Blackwell 兼容 wheel，CUDA 12.8+ 运行时。
- 训练：Diffusers、Transformers、Accelerate、PEFT、Datasets、Safetensors、TensorBoard、bitsandbytes（WSL/Linux）。
- 数据：Pillow、NumPy、OpenCV、tqdm；Depth/Caption 模型在阶段 C 再按实际方案加入，避免骨架安装就下载重依赖。
- 推理/UI：Diffusers、Gradio。
- 评测：open-clip-torch、TorchMetrics、pandas。
- 开发：pytest、pytest-cov、Ruff。

版本范围在 `pyproject.toml` 中有上下界。大模型 revision 和训练运行时的 `pip freeze` 将写入实验元数据，避免只依赖浮动库版本重现结果。
