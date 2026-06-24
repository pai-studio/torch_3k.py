# PLAN-008 优化器状态保存与恢复结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-008-optimizer-state-dict.md`

Git 基线：`6290dc357a5588585be2ddba989b3d078a4e4bf8`

## 实施结果

1. `Optimizer` 基类新增状态辅助协议：
   - 参数对象 `id` 到稳定参数序号的映射。
   - 单参数组 `param_groups` 生成与加载校验。
   - 状态数组保存为独立 `Tensor` 拷贝。
   - 加载时按目标参数 device 拷贝状态数组。
2. `SGD` 新增：
   - `state_dict()`
   - `load_state_dict()`
   - 保存和恢复 `lr`。
3. `MomentumSGD` 新增：
   - 保存和恢复 `lr`、`momentum`。
   - 保存和恢复每个参数的 `velocity`。
4. `Adam` 新增：
   - 保存和恢复 `lr`、`betas`、`eps`、`weight_decay`。
   - 保存和恢复全局步数 `t`。
   - 保存和恢复每个参数的 `m` / `v` 一二阶矩。
5. `AdamW` 新增：
   - 保存和恢复 decoupled `weight_decay`。
   - 加载后保持 `AdamW` 的解耦权重衰减语义。
6. 更新设计文档：
   - 记录 PLAN-008 已补齐优化器状态保存。
   - 将当前风险改为“尚未支持 PyTorch 完整多 param group 体系”。

## 新增测试

新增 `tests/test_17_optimizer_state.py`，覆盖：

1. Adam checkpoint 恢复：
   - 先训练 6 步。
   - 保存 `model.state_dict()` 和 `optimizer.state_dict()`。
   - 原模型继续训练 4 步。
   - 新模型加载模型和优化器状态后继续训练 4 步。
   - 两条路径输出一致。
2. SGD 状态恢复 `lr`。
3. MomentumSGD 状态恢复 `lr`、`momentum` 和 `velocity`。

## 验证结果

已运行：

```bash
pytest -q tests/test_17_optimizer_state.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增优化器状态测试：`3 passed`
- 全量测试：`57 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## 复核结论

本轮实现能支撑现有模型状态与优化器状态一起组成 checkpoint。状态文件使用参数序号保存，不依赖原优化器实例中的 `id(parameter)`，因此可以加载到同结构的新模型优化器上。当前实现仍只覆盖单参数组；多 param group、scheduler 状态和更完整的 PyTorch optimizer 语义属于后续扩展范围。
