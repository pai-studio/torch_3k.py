# PLAN-014 nn.init 初始化工具

日期：2026-06-24

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 目标

补齐真实模型定义中常用的 `torch.nn.init` 子集，使用户可以在 `torch_1k` 中用教学友好的方式控制参数初始值，并保持 CPU/CUDA 后端一致。

本计划只覆盖核心初始化函数，不追求完整 PyTorch 初始化 API。

## 当前缺口

1. 路线图将“简单初始化工具”列为 `nn` 层后续核心模块，但当前 `torch_1k.nn` 没有 `init` 子模块。
2. 用户无法写 `nn.init.xavier_uniform_(layer.weight)` 或 `nn.init.zeros_(layer.bias)` 这类常见模型初始化代码。
3. `Linear` 权重布局是本库教学实现的 `(in_features, out_features)`，与 PyTorch 的 `(out_features, in_features)` 不同；如果直接按 PyTorch 原始 shape 推断 fan，`kaiming_uniform_` 的 `fan_in` 会不符合本库前向语义。
4. 初始化工具需要使用当前 Tensor 所在后端，避免 CUDA Tensor 初始化后退回 CPU。

## 实施步骤

1. 新增 `torch_1k/nn/init.py`：
   - `constant_(tensor, val)`
   - `zeros_(tensor)`
   - `ones_(tensor)`
   - `uniform_(tensor, a=0.0, b=1.0)`
   - `normal_(tensor, mean=0.0, std=1.0)`
   - `xavier_uniform_(tensor, gain=1.0)`
   - `kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`
2. 初始化函数全部就地修改 `tensor.data`，并返回原 Tensor。
3. 使用 `backend.get_array_module(tensor.data)` 选择 NumPy 或 CuPy。
4. 为 `Linear.weight` 和 `Conv2d.weight` 记录 `_fan_in` / `_fan_out` 元数据，使 fan 计算符合本库前向语义。
5. 在 `torch_1k.nn.__init__` 导出 `init` 子模块，支持 `import torch_1k.nn as nn; nn.init.zeros_(...)`。
6. 新增 `examples/example14_nn_init_compare.py`，覆盖常见初始化写法。
7. 新增 `tests/test_23_nn_init.py`，覆盖：
   - 常量、全零、全一初始化。
   - uniform/normal 就地返回同一 Tensor。
   - Xavier/Kaiming bound 与 fan 元数据。
   - CUDA Tensor 可选路径。
8. 更新设计文档记录 `nn.init` 阶段完成。
9. 运行新增测试、示例、全量测试和编译检查。

## 验收标准

1. `nn.init.zeros_(linear.bias)` 可以就地置零并返回原 Tensor。
2. `nn.init.xavier_uniform_(linear.weight)` 使用正确 fan 边界。
3. `nn.init.kaiming_uniform_(linear.weight, mode='fan_in')` 对本库 `Linear(in, out)` 使用 `in_features` 作为 fan_in。
4. CUDA Tensor 初始化后仍保持 CUDA 设备。
5. 全量测试通过。

## TODO 记录

1. 本计划不实现 `xavier_normal_`、`kaiming_normal_`、`orthogonal_`、`trunc_normal_` 等较完整初始化 API。
2. 本计划不改变 `Linear` 权重布局，避免引入大范围行为迁移。
