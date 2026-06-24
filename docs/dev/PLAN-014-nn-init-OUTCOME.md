# PLAN-014 nn.init 初始化工具结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-014-nn-init.md`

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 实施结果

1. 新增 `torch_1k.nn.init` 子模块：
   - `constant_(tensor, val)`
   - `zeros_(tensor)`
   - `ones_(tensor)`
   - `uniform_(tensor, a=0.0, b=1.0)`
   - `normal_(tensor, mean=0.0, std=1.0)`
   - `xavier_uniform_(tensor, gain=1.0)`
   - `kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`
   - `calculate_gain(nonlinearity, param=None)`
2. 所有初始化函数均直接就地修改 `tensor.data`，并返回原 Tensor。
3. 初始化函数通过 `backend.get_array_module(tensor.data)` 选择 NumPy 或 CuPy，避免 CUDA Tensor 初始化后退回 CPU。
4. `torch_1k.nn.__init__` 已导出 `init` 子模块，支持：

```python
import torch_1k.nn as nn
nn.init.xavier_uniform_(layer.weight)
```

5. `Linear.weight` 记录 `_fan_in` / `_fan_out`：
   - 本库 `Linear` 权重布局是 `(in_features, out_features)`。
   - `kaiming_uniform_(mode='fan_in')` 会使用真实前向语义中的 `in_features`。
6. `Conv2d.weight` 记录 `_fan_in` / `_fan_out`：
   - fan 计算为 `in_channels * kh * kw` 和 `out_channels * kh * kw`。
7. 新增 `examples/example14_nn_init_compare.py`：
   - 覆盖 `xavier_uniform_`、`kaiming_uniform_`、`zeros_`、`ones_`。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
8. 更新设计文档，记录 `nn.init` 常用子集已补齐。

## 新增测试

新增 `tests/test_23_nn_init.py`，覆盖：

1. `constant_` / `zeros_` / `ones_` 就地修改并返回原 Tensor。
2. `uniform_` / `normal_` 的随机填充和返回值。
3. `xavier_uniform_` 使用 `Linear` fan 元数据计算边界。
4. `kaiming_uniform_` 使用 `Linear` fan_in 元数据。
5. `kaiming_uniform_` 使用 `Conv2d` fan_in 元数据。
6. 非 Tensor 输入明确报错。
7. 非法 `mode` 明确报错。
8. 可选 CUDA Tensor 初始化后保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_23_nn_init.py
python examples/example14_nn_init_compare.py
USE_TORCH_1K=0 python examples/example14_nn_init_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 nn.init 测试：`8 passed`
- nn.init 双后端示例：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`90 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 `xavier_normal_`、`kaiming_normal_`、`orthogonal_`、`trunc_normal_` 等完整初始化 API。
2. 本轮没有改变 `Linear` 权重布局，后续如要更完整贴近 PyTorch 参数布局，需要单独设计迁移。

## 复核结论

本轮补齐了真实模型定义中常用的初始化工具，并保持教学实现的简单边界。`nn.init` 现在覆盖常见常量、均匀、正态、Xavier 和 Kaiming 初始化，且 CPU/CUDA 后端共享同一套实现。
