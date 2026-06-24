# PLAN-007 常用函数式 API 兼容结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-007-common-functional-api.md`

Git 基线：`d8c16cd3dfca4b28cbf504705946c8882271b67f`

## 实施结果

1. 新增统一函数式算子：
   - `torch_1k.log(x)`
   - `torch_1k.relu(x)`
2. 新增 Tensor 方法：
   - `x.exp()`
   - `x.log()`
   - `x.relu()`
   - `x.tanh()`
   - `x.sigmoid()`
3. `torch_1k.nn.functional.relu` 改为复用 `torch_1k.functional.numeric.ReLU`，避免顶层函数和 nn functional 维护两份反向传播实现。
4. 新增 `examples/example8_functional_api_compare.py`：
   - 覆盖 `torch.relu`、`x.log()` 和 `nn.functional.relu`。
   - 同一代码可通过 `USE_TORCH_1K=0` 切到 PyTorch。
5. 更新 README 和设计文档：
   - README 将常用函数 `sin/cos/exp/log/relu/softmax` 标记为已完成。
   - 设计文档记录 PLAN-007 的函数式 API 阶段状态。
   - 清理 `torch-1k` 原理文档中过时的已修复风险列表。

## 新增测试

新增 `tests/test_16_functional_api.py`，覆盖：

1. `torch.log(x)` 与 `x.log()` 的前向值和反向梯度。
2. `torch.relu(x)`、`x.relu()` 与 `nn.functional.relu(x)` 的前向值和反向梯度。
3. `x.exp()`、`x.tanh()`、`x.sigmoid()` 的方法形式前向结果。

## 验证结果

已运行：

```bash
pytest -q tests/test_16_functional_api.py
python examples/example8_functional_api_compare.py
USE_TORCH_1K=0 python examples/example8_functional_api_compare.py
python -m compileall -q torch_1k examples
pytest -q
```

结果：

- 新增函数式 API 测试：`3 passed`
- 函数式示例：`torch_1k` 与 PyTorch 均输出 `score=0.393919`
- 全量测试：`54 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## 复核结论

本轮变更保持在常用函数式 API 兼容范围内，没有引入新的依赖。`torch.log` 对非正输入仍遵循底层 NumPy/CuPy 的数值行为，和 PyTorch 一样由用户保证输入域；后续更高价值目标是补齐 `requires_grad` 或优化器状态保存，减少真实训练和 checkpoint 工作流中的兼容差距。
