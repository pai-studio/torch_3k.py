# PLAN-028 CrossEntropyLoss 高级常用参数结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-028-cross-entropy-advanced.md`

Git 基线：`81ffeb1`

## 实施结果

1. `nn.CrossEntropyLoss` 新增常用高级参数：
   - `weight=None`
   - `ignore_index=-100`
   - `reduction="mean"`
   - `label_smoothing=0.0`
2. `nn.functional.cross_entropy(...)` 同步支持上述参数。
3. 支持二维分类 logits：
   - input: `(N, C)`
   - target: `(N,)`
4. 支持分割/序列风格高维 logits：
   - input: `(N, C, d1, d2, ...)`
   - target: `(N, d1, d2, ...)`
5. `ignore_index` 位置现在：
   - 前向不计入损失。
   - `reduction="none"` 返回 0。
   - 反向传播对该位置 logits 梯度为 0。
6. `weight` 现在按目标类别加权 NLL，`mean` 按非忽略目标类别权重和归一化。
7. `label_smoothing` 支持与 `weight`、`ignore_index`、三种 reduction 组合使用。
8. 新增参数和 target/weight shape 校验，避免底层索引错误泄漏到用户接口。

## 新增测试

新增 `tests/test_36_cross_entropy_advanced.py`，覆盖：

1. 高维 `(N, C, H, W)` 输入与 PyTorch 前向对比。
2. `weight`、`ignore_index`、`label_smoothing` 组合与 PyTorch 梯度对比。
3. `"none"`、`"sum"`、`"mean"` 三种 reduction。
4. `ignore_index` 在 `reduction="none"` 下返回 0 且梯度为 0。
5. `nn.CrossEntropyLoss` 与 `nn.functional.cross_entropy` 高级参数行为一致。
6. 非法 target shape、weight shape、target 类别和 label_smoothing 报错。
7. 可选 CUDA 路径下高级参数输出和梯度保持 CUDA 设备。

## 新增示例

新增 `examples/example28_cross_entropy_advanced_compare.py`：

1. 在同一进程中分别运行 `torch_1k` 与 PyTorch。
2. 使用分割风格 logits `(N, C, H, W)`。
3. 同时启用类别权重、`ignore_index` 和 `label_smoothing`。
4. 对比 `loss_none`、`loss_sum`、`loss_mean` 和 logits 梯度。

## 验证结果

已运行：

```bash
pytest -q tests/test_30_loss_functional_reduction.py
pytest -q tests/test_36_cross_entropy_advanced.py
pytest -q tests/test_30_loss_functional_reduction.py tests/test_36_cross_entropy_advanced.py tests/test_12_mlp.py tests/test_13_cnn_transformer.py tests/test_17_optimizer_state.py
python examples/example21_loss_functional_reduction_compare.py
USE_TORCH_1K=0 python examples/example21_loss_functional_reduction_compare.py
python examples/example28_cross_entropy_advanced_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 既有损失函数 reduction 测试：`6 passed`
- 新增高级交叉熵测试：`7 passed`
- 损失和训练链路相关测试：`21 passed`
- 既有损失示例：`torch_1k` 与 PyTorch 路径均通过
- 新增同进程 PyTorch 对比示例：通过，最大梯度差 `2.775557561563e-17`
- 全量测试：`190 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 仍不支持 soft target / probability target 形式。
2. 仍不让 `weight` 参与反向传播。
3. 仍不实现完整 dtype promotion。

## 复核结论

本轮把交叉熵从基础分类损失扩展到真实训练常见形态。现在类别不均衡分类、NLP padding token、语义分割 void label 和 label smoothing 都能使用接近 PyTorch 的写法表达，并且 CPU/CUDA 后端共用同一套计算路径。
