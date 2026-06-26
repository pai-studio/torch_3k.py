# PLAN-038 新手教程驱动基础 API 加固

日期：2026-06-26

Git 基线：`6b1c9bb`

设计文档：`docs/design/torch-tutorial-20260626-api-dependency.md`

## 背景

如果给 torch 新手写一组入门例子，教学路径通常不是从单个 API 开始，而是从“张量基础 -> 自动微分 -> 手写模型 -> nn.Module -> DataLoader -> CNN/Transformer 方向感”逐层展开。PLAN-036 / PLAN-037 已经证明训练脚本和入口层能跑通，但教程场景里还会自然用到一些基础 wrapper：

1. `torch.eye`
2. `torch.as_tensor`
3. `torch.from_numpy`
4. `torch.randn_like` / `torch.rand_like`
5. `Tensor.clone`
6. `Tensor.tolist`
7. `Tensor.matmul` / `Tensor.mm`
8. `torch.sum(..., dim=..., keepdim=...)`
9. `torch.mean(..., dim=..., keepdim=...)`
10. `nn.Tanh` / `nn.Sigmoid`

这些能力有的只是 wrapper，有的需要底层 Function 支撑，例如 `clone` 应保持可微。补齐它们可以让教程代码更自然，也能巩固基础层。

## 目标

1. 新增教程式多场景例子：
   - 张量基础与 NumPy 转换。
   - 自动微分。
   - 手写线性回归。
   - `nn.Module` MLP 分类。
   - `DataLoader` mini-batch 训练。
2. 新增/扩展基础 API：
   - `torch_1k.eye`
   - `torch_1k.as_tensor`
   - `torch_1k.from_numpy`
   - `torch_1k.rand_like`
   - `torch_1k.randn_like`
   - `Tensor.clone`
   - `Tensor.tolist`
   - `Tensor.matmul`
   - `Tensor.mm`
   - `Tensor.__len__`
   - `torch_1k.sum(input, dim=None, keepdim=False)`
   - `torch_1k.mean(input, dim=None, keepdim=False)`
   - `Tensor.sum(..., keepdim=...)`
   - `Tensor.mean(..., keepdim=...)`
   - `nn.Tanh`
   - `nn.Sigmoid`
3. 新增测试：
   - 与 PyTorch 对比基础 wrapper 前向和关键梯度。
   - 运行教程例子默认 `torch_1k` 路径。
   - 运行 `USE_TORCH_1K=0` PyTorch 替换路径。
4. 更新路线图和结果文档。

## 实施步骤

1. 改造 `torch_1k/functional/matrix.py`：
   - 新增可微 `clone` Function。
2. 改造 `torch_1k/tensor.py`：
   - 扩展 `sum` / `mean` 的 `keepdim` 别名。
   - 新增 `clone`、`tolist`、`matmul`、`mm`、`__len__`。
   - 新增 `eye`、`as_tensor`、`from_numpy`、`rand_like`、`randn_like`。
3. 改造 `torch_1k/misc.py`：
   - 新增顶层 `sum` / `mean` PyTorch 风格 wrapper。
   - 为 `linspace` / `normal` 补常见 `dtype` 参数。
4. 改造 `torch_1k/nn/activation.py` 和导出：
   - 新增 `Tanh`、`Sigmoid`。
5. 新增 `examples/example38_beginner_tutorial.py`。
6. 新增 `tests/test_44_beginner_tutorial_core_api.py`。
7. 更新 `docs/design/torch-3k-20260624-core-cuda-roadmap.md` 和 OUTCOME。

## 非目标

1. 不实现完整 dtype promotion。
2. 不实现原地操作族。
3. 不实现完整高级索引。
4. 不实现 torchvision 风格数据集和 transform。
5. 不把所有 PyTorch 教程 API 一次补齐；只补本轮教程例子直接依赖的基础能力。

## 验收标准

1. 新增基础 API 测试通过。
2. 新增教程例子在 `torch_1k` 路径通过。
3. 新增教程例子在 PyTorch 路径通过，或未安装 PyTorch 时测试明确跳过。
4. PLAN-036 / PLAN-037 基线不回归。
5. 全量测试通过。
6. `python -m compileall -q torch_1k examples` 通过。
7. OUTCOME 记录实现范围、验证结果、结果 Git 记录和未完成事项。
