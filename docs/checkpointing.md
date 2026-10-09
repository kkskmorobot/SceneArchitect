# 训练中断存档与恢复契约

## 保存内容

`Accelerator.save_state()` 负责模型、优化器、调度器、AMP scaler、RNG 与已注册对象；`TrainingProgress` 作为注册对象保存 global step、epoch、epoch 内 batch、样本数和最佳指标。SceneArchitect 额外保存：

- 配置 SHA-256 指纹；
- 创建时间、保存原因与运行 ID；
- Python/平台信息；
- `manifest.json`：每个状态文件的相对路径、字节数和 SHA-256；
- `_SUCCESS` 完整性标记。

单独导出的 `pytorch_lora_weights.safetensors` 适合推理，不足以无缝续训，因此不能替代完整 checkpoint。

## 原子性

状态先写到同一父目录的隐藏 `.tmp` 目录。全部文件、metadata 和 manifest 成功后写 `_SUCCESS`，再重命名为正式目录。`latest` 只搜索符合命名规则、带 `_SUCCESS` 且 manifest 文件齐全的目录；正式恢复还会验证所有 SHA-256，因此截断、删改或安全软件破坏的状态不会被静默读取。

## 中断行为

`StopController` 捕捉第一次 SIGINT/SIGTERM，只设置停止请求；训练循环在当前 optimizer step 完成后调用 `raise_if_requested()`。外围 `emergency_checkpoint_guard` 捕捉安全停止、Ctrl+C 或普通异常，并根据配置尝试保存。第二次信号允许立即终止。

GPU/磁盘故障可能同时让紧急保存失败；代码必须记录原始异常和保存异常，不能声称一定保存成功。周期 checkpoint 仍是主要保护。

## 恢复顺序

1. 创建与原训练相同的模型、LoRA adapter、优化器和 scheduler。
2. 用 `accelerator.prepare(...)` 包装对象。
3. 向 accelerator 注册 `TrainingProgress`。
4. `CheckpointManager.restore(accelerator, "latest", current_config=...)`。
5. 从恢复后的 `TrainingProgress` 继续 dataloader skip 与 global step。

`strict_config_match: true` 时配置指纹不同会拒绝恢复。若确需迁移，必须新建实验目录或显式关闭严格模式并在日志中说明，不能静默混用。
