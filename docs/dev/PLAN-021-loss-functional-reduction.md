# PLAN-021 损失函数 reduction 与函数式 API

日期：2026-06-25

Git 基线：`a5d4d3b`

## 背景

本库当前定位是用教学友好的最简代码实现 PyTorch 核心训练链路。现有 `nn.CrossEntropyLoss` 和 `nn.MSELoss` 已能支撑基础示例，但仍只覆盖默认 `mean` 行为，且 `torch_1k.nn.functional` 尚未提供 `cross_entropy` / `mse_loss`。

真实训练代码中，分类任务常见写法包括：

```python
loss = nn.CrossEntropyLoss(reduction="sum")(logits, target)
loss = F.cross_entropy(logits, target, reduction="none")
```

回归任务也常用：

```python
loss = F.mse_loss(pred, target, reduction="mean")
```

因此，本轮优先补齐损失函数 reduction 和函数式入口，提升 PyTorch 替换兼容性，同时保持实现短小、可读、CPU/CUDA 后端一致。

## 目标

1. 为 `nn.CrossEntropyLoss` 支持 `reduction="none" | "mean" | "sum"`。
2. 为 `nn.MSELoss` 支持 `reduction="none" | "mean" | "sum"`。
3. 新增 `torch_1k.nn.functional.cross_entropy(...)`。
4. 新增 `torch_1k.nn.functional.mse_loss(...)`。
5. 保持现有默认行为不变：未指定 `reduction` 时仍为 `"mean"`。
6. 保持 CUDA/CuPy 路径可用：输入在 CUDA 上时，输出和梯度也应保留在 CUDA 后端。

## 实施步骤

1. 重构 `torch_1k/nn/loss.py`：
   - 增加 reduction 校验函数。
   - 修正 `MSELoss.__init__` 调用基类的方式。
   - 在前向和反向中按 reduction 处理损失值和梯度缩放。
2. 更新 `torch_1k/nn/functional.py`：
   - 暴露 `cross_entropy(input, target, reduction="mean")`。
   - 暴露 `mse_loss(input, target, reduction="mean")`。
3. 增加测试：
   - 覆盖 `CrossEntropyLoss` 三种 reduction 的前向值。
   - 覆盖 `cross_entropy` 的梯度缩放。
   - 覆盖 `mse_loss` 的函数式入口和 target 梯度。
   - 覆盖非法 reduction 报错。
   - 增加可选 CUDA 测试。
4. 增加 PyTorch 兼容示例：
   - 同一份示例通过 `USE_TORCH_1K=0/1` 切换 PyTorch 与 torch_1k。
5. 更新设计文档阶段记录。

## 非目标

1. 不实现 `CrossEntropyLoss` 的 `weight`、`ignore_index`、`label_smoothing`。
2. 不实现高维分割任务形态的交叉熵输入，例如 `(N, C, H, W)`。
3. 不实现完整 dtype promotion。
4. 不改变现有训练示例的默认行为。

## 验收标准

1. 新增测试通过。
2. 现有测试不回归。
3. 新增示例在 `torch_1k` 与 PyTorch 两条路径下通过。
4. `python -m compileall -q torch_1k examples` 通过。
5. 结果文档记录实现范围、验证结果和未完成事项。
