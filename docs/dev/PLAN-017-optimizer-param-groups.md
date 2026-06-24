# PLAN-017 优化器多参数组

日期：2026-06-24

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 目标

补齐真实训练中常用的 optimizer param group 能力，使用户可以为不同参数集合配置不同学习率、动量、Adam betas、eps 和权重衰减。

本计划聚焦已有优化器的多参数组支持，不引入 scheduler 或完整 PyTorch Optimizer API。

## 当前缺口

1. 设计文档记录当前优化器只支持单参数组。
2. `Optimizer._validate_state_dict()` 明确要求 `param_groups` 长度为 1。
3. 真实训练中常见配置无法表达，例如：

```python
optim.Adam([
    {"params": encoder.parameters(), "lr": 1e-4},
    {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-2},
])
```

4. 当前 `state_dict()` 不能保存多个参数组的超参数。

## 实施步骤

1. 改造 `Optimizer`：
   - 支持传入普通参数 iterable。
   - 支持传入 param group dict 列表。
   - 保存 `self.param_groups`。
   - 扁平化维护 `self.parameters`，保持现有代码兼容。
   - `step()` 按 group 调用 `update_one(parameter, group)`。
   - `state_dict()` 输出多组 `param_groups`。
   - `load_state_dict()` 校验组数和每组参数数量。
2. 适配 `SGD`：
   - 每组可配置 `lr`。
3. 适配 `MomentumSGD`：
   - 每组可配置 `lr` / `momentum`。
4. 适配 `Adam`：
   - 每组可配置 `lr` / `betas` / `eps` / `weight_decay`。
5. 适配 `AdamW`：
   - 每组可配置 `lr` / `betas` / `eps` / decoupled `weight_decay`。
6. 保留旧属性：
   - 单参数组场景下 `optimizer.lr`、`optimizer.momentum` 等仍反映第一组配置。
7. 新增 `examples/example17_optimizer_param_groups_compare.py`，覆盖 PyTorch import 切换路径。
8. 新增 `tests/test_26_optimizer_param_groups.py`，覆盖：
   - SGD 不同参数组不同 lr。
   - MomentumSGD 不同组 momentum。
   - Adam 不同组 lr/weight_decay。
   - AdamW decoupled weight decay。
   - 多参数组 state_dict/load_state_dict 恢复。
   - CUDA 参数可选路径。
9. 更新设计文档记录 PLAN-017。
10. 运行新增测试、示例、全量测试和编译检查。

## 验收标准

1. `optim.SGD([{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.01}])` 可用。
2. `optim.Adam([...])` 支持每组不同 `lr` 和 `weight_decay`。
3. `optimizer.state_dict()["param_groups"]` 保留多组超参数。
4. `load_state_dict()` 可恢复多组超参数和已有动量/Adam 状态。
5. 现有单参数组 optimizer 测试继续通过。
6. CUDA Tensor 参数仍可更新。
7. 全量测试通过。

## TODO 记录

1. 本计划不实现 scheduler。
2. 本计划不实现 PyTorch optimizer hook、foreach/fused、maximize、capturable 等完整选项。
3. 本计划不实现参数重复检测。
