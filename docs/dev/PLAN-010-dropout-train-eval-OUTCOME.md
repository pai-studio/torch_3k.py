# PLAN-010 Dropout 与 train/eval 行为结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-010-dropout-train-eval.md`

Git 基线：`9f4acc33fc98b2980f081d6b80c6708853ba80ae`

## 实施结果

1. 新增 `torch_1k.nn.functional.dropout(input, p=0.5, training=True, inplace=False)`：
   - 校验 `p` 必须在 `[0, 1]`。
   - `training=False` 或 `p=0` 时返回输入。
   - `p=1` 时返回全零输出并保留梯度路径。
   - 训练态使用 inverted dropout：保留元素按 `1 / (1 - p)` 缩放。
2. 新增 `torch_1k.nn.Dropout`：
   - 保存 `p` 和 `inplace` 参数。
   - forward 根据模块 `self.training` 调用函数式 dropout。
3. 从 `torch_1k.nn` 导出 `Dropout`。
4. 新增 `examples/example10_dropout_train_eval_compare.py`：
   - 验证训练态会产生零值和缩放值。
   - 验证 eval 后输出等于输入。
   - 验证反向梯度等于训练态 mask 缩放值。
   - 同一代码支持 `torch_1k` 与 PyTorch 双路径。
5. 更新 README 和设计文档，记录 Dropout 已完成，并将剩余 nn 边界调整为 BatchNorm、初始化工具和更完整容器模块。

## 新增测试

新增 `tests/test_19_dropout.py`，覆盖：

1. 训练态随机 mask、缩放值和反向梯度。
2. eval 模式恒等输出并保持 `requires_grad`。
3. `nn.functional.dropout` 的 `training`、`p=0`、`p=1` 边界。
4. 非法 `p` 参数校验。

## 验证结果

已运行：

```bash
pytest -q tests/test_19_dropout.py
python examples/example10_dropout_train_eval_compare.py
USE_TORCH_1K=0 python examples/example10_dropout_train_eval_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 Dropout 测试：`4 passed`
- Dropout 双后端示例：`torch_1k` 与 PyTorch 均通过
- 全量测试：`67 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## 复核结论

本轮变更让 `train()` / `eval()` 具备可观察的模块行为差异。当前 `inplace` 参数仅保留兼容签名，不执行原地修改；这避免破坏现有动态图实现。后续高价值目标可以继续补齐 BatchNorm，或转向 `DataLoader` sampler / 多 param group 等更完整 PyTorch 兼容面。
