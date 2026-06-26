# PLAN-021 损失函数 reduction 与函数式 API 结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-021-loss-functional-reduction.md`

Git 基线：`a5d4d3b`

## 实施结果

1. `nn.CrossEntropyLoss` 新增 `reduction` 参数：
   - `reduction="mean"`：保持原有默认行为。
   - `reduction="sum"`：返回 batch 损失总和，反向传播不按 batch 平均缩放。
   - `reduction="none"`：返回每个样本的损失向量，反向传播按上游梯度逐样本缩放。
2. `nn.MSELoss` 新增 `reduction` 参数：
   - 支持 `"none"`、`"mean"`、`"sum"`。
   - 修正原先 `MSELoss.__init__` 调用基类方式不规范的问题。
   - 反向传播现在同时支持 input 和 target 的梯度，并处理广播形状还原。
3. `torch_1k.nn.functional` 新增函数式入口：
   - `cross_entropy(input, target, reduction="mean")`
   - `mse_loss(input, target, reduction="mean")`
4. 新增 `examples/example21_loss_functional_reduction_compare.py`：
   - 覆盖 `F.cross_entropy`、`nn.CrossEntropyLoss(reduction=...)`。
   - 覆盖 `F.mse_loss`、`nn.MSELoss(reduction=...)`。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
5. 更新设计文档，记录 PLAN-021 已补齐损失函数 reduction 与函数式 API。

## 新增测试

新增 `tests/test_30_loss_functional_reduction.py`，覆盖：

1. `F.cross_entropy` 三种 reduction 的前向值。
2. `F.cross_entropy` 在 `"sum"`、`"mean"`、`"none"` 下的梯度缩放。
3. `nn.CrossEntropyLoss(reduction="none")` 的模块式入口。
4. `F.mse_loss` 三种 reduction 的前向值。
5. `F.mse_loss` 对 input 和 target 的梯度。
6. 非法 reduction 报错。
7. 可选 CUDA 路径下 `cross_entropy` 输出和梯度保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_30_loss_functional_reduction.py
python examples/example21_loss_functional_reduction_compare.py
USE_TORCH_1K=0 python examples/example21_loss_functional_reduction_compare.py
pytest -q tests/test_12_mlp.py tests/test_13_cnn_transformer.py tests/test_17_optimizer_state.py
python examples/example4_mlp_train_compare.py
USE_TORCH_1K=0 python examples/example4_mlp_train_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增损失函数测试：`6 passed`
- 新增损失函数示例：`torch_1k` 与 PyTorch 路径均通过
- 旧训练链路相关测试：`8 passed`
- XOR MLP 示例：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`136 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. `CrossEntropyLoss` 的 `weight`、`ignore_index`、`label_smoothing` 已在 PLAN-028 补齐。
2. 高维 `(N, C, ...)` 交叉熵输入形态已在 PLAN-028 补齐。
3. 尚未实现完整 dtype promotion。

## 复核结论

本轮补齐了分类和回归训练中最常见的损失函数写法。现在默认训练示例保持不变，同时可以用 PyTorch 风格的函数式入口和 reduction 参数表达按样本损失、总和损失与平均损失，CPU/CUDA 后端路径保持一致。
