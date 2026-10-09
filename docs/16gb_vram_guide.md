# 单卡 16GB 显存策略

## 默认训练档

- SDXL base，冻结 VAE 与两个文本编码器。
- 仅 UNet LoRA，rank 16 / alpha 16。
- bf16；batch size 1；gradient accumulation 4。
- 768 中心/随机裁剪与 aspect ratio buckets，最大像素面积约 `768 x 768`。
- gradient checkpointing 开启；PyTorch SDPA；WSL2/Linux 上 8-bit AdamW。
- 验证一次只生成一张图，验证后主动释放 pipeline 和 CUDA cache。

## 从 768 升到 1024

1. 先用 512/640 的 20-step smoke test 验证数据与断点链路。
2. 正式通用 LoRA 在 768 分桶训练并观察峰值显存。
3. 只在数据质量稳定后新建 1024 配置；保持 batch 1，降低 rank 或关闭随机增强，必要时使用磁盘 latent cache。
4. 1024 训练作为短程第二阶段，使用更低学习率并保留 768 结果，不覆盖原实验。
5. 即使不做 1024 LoRA 训练，SDXL 推理仍可输出 1024；训练分辨率不等于最终输出硬上限。

## 推理档

- 文本或单控制：1024，bf16，SDPA，VAE slicing/tiling；视实测决定整管线 GPU 或 CPU offload。
- 双控制：先用 768，`model_cpu_offload: true`，ControlNet 数量上限 2，batch 1。
- 高分辨率：先完成结构稳定的基础图，再用低 strength img2img/tile refine；不要默认直接 2048。

## OOM 降配顺序

1. batch 固定为 1；生成张数改为串行。
2. 双控制降为单控制，或把尺寸从 1024 降到 896/768。
3. 开启 model CPU offload、VAE tiling/slicing；关闭同时驻留的可选模型。
4. 训练关闭文本编码器、降低 LoRA rank、使用 8-bit optimizer。
5. 训练像素面积降到 704/640，并提高梯度累积保持有效 batch。
6. 最后才考虑 sequential CPU offload；它通常最慢。

所有“能否恰好放进 16GB”的数字都受 PyTorch、驱动、模型 checkpoint、注意力后端和分辨率形状影响，正式训练前必须用目标环境记录 `torch.cuda.max_memory_allocated()` 做 smoke test。

