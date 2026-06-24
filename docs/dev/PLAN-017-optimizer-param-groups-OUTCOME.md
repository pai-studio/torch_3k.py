# PLAN-017 优化器多参数组结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-017-optimizer-param-groups.md`

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 实施结果

1. `Optimizer` 基类新增 param group 机制：
   - 支持普通参数 iterable。
   - 支持 param group dict 列表。
   - 维护 `self.param_groups`。
   - 保留扁平化 `self.parameters`，兼容既有代码。
   - `step()` 按 group 分发到 `update_one(parameter, group)`。
2. `state_dict()` 支持多参数组：
   - 每个 group 保存自己的超参数。
   - 参数仍以扁平化索引保存，便于加载到同结构新模型。
3. `load_state_dict()` 支持多参数组：
   - 校验参数组数量。
   - 校验每组参数数量。
   - 恢复每组超参数。
4. `SGD` 支持每组不同 `lr`。
5. `MomentumSGD` 支持每组不同 `lr` / `momentum`。
6. `Adam` 支持每组不同：
   - `lr`
   - `betas`
   - `eps`
   - coupled `weight_decay`
7. `AdamW` 支持每组不同：
   - `lr`
   - `betas`
   - `eps`
   - decoupled `weight_decay`
8. 保留旧属性兼容：
   - `optimizer.lr`
   - `optimizer.momentum`
   - `optimizer.beta1` / `optimizer.beta2`
   - `optimizer.eps`
   - `optimizer.weight_decay`
   - `optimizer.decoupled_weight_decay`

这些属性反映第一组配置，保持现有单参数组测试和用户代码可用。

9. 新增 `examples/example17_optimizer_param_groups_compare.py`：
   - 覆盖不同参数组不同学习率。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
10. 更新设计文档，记录已有优化器已补齐多参数组支持。

## 新增测试

新增 `tests/test_26_optimizer_param_groups.py`，覆盖：

1. `SGD` 两个参数组使用不同 `lr`。
2. `MomentumSGD` 两个参数组使用不同 `momentum`。
3. `Adam` 两个参数组使用不同 `lr` / `weight_decay`。
4. `AdamW` 使用每组 decoupled `weight_decay`。
5. 多参数组 `state_dict()` / `load_state_dict()` 恢复超参数和 Adam 状态。
6. 参数组数量不匹配时 `load_state_dict()` 报错。
7. 可选 CUDA 参数组路径。

## 验证结果

已运行：

```bash
pytest -q tests/test_26_optimizer_param_groups.py
python examples/example17_optimizer_param_groups_compare.py
USE_TORCH_1K=0 python examples/example17_optimizer_param_groups_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 optimizer param group 测试：`7 passed`
- optimizer param group 双后端示例：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`112 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 scheduler。
2. 尚未实现 PyTorch optimizer hook、foreach/fused、maximize、capturable 等完整选项。
3. 尚未实现参数重复检测。

## 复核结论

本轮补齐了真实训练中常见的分组优化配置能力。现在已有优化器可以表达不同层不同学习率、动量、Adam 超参数和权重衰减，并且 checkpoint 能保存恢复多参数组状态。
