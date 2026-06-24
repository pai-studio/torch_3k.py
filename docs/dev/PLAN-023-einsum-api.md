# PLAN-023 einsum 核心算子

日期：2026-06-25

Git 基线：`a5d4d3b`

说明：当前工作区已包含 PLAN-021 和 PLAN-022 的未提交改动，本计划在该工作区状态上继续实施。

## 背景

爱因斯坦求和记号是深度学习中常见且高价值的张量算子。一个 `einsum` 可以清晰表达矩阵乘、批量矩阵乘、注意力打分、逐元素乘加和多维规约。对教学型 PyTorch 核心实现而言，`einsum` 很适合作为“从数学记号到自动微分”的示例算子。

真实代码中常见用法包括：

```python
torch.einsum("ij,jk->ik", x, w)
torch.einsum("bqd,bkd->bqk", query, key)
torch.einsum("bqk,bkd->bqd", attn, value)
torch.einsum("ij->i", x)
```

本轮实现一个可读、稳定、覆盖常用训练场景的核心子集。

## 目标

1. 新增顶层 `torch_1k.einsum(equation, *operands)`。
2. 支持显式输出，例如 `"ij,jk->ik"`。
3. 支持隐式输出，例如 `"ij,jk"`。
4. 支持多个输入。
5. 支持输出为标量的规约，例如 `"ij->"`。
6. 支持常见反向传播：
   - 矩阵乘。
   - batch matmul。
   - attention score / context。
   - 单输入规约。
7. 保持 CPU/CUDA 后端一致：前向和反向都使用输入所在后端的 `einsum`。

## 实施步骤

1. 在 `torch_1k/functional/matrix.py` 中实现：
   - equation 解析和校验。
   - `Einsum(Function)`。
   - `einsum(equation, *operands)` 顶层函数。
2. 反向传播策略：
   - 对每个需要梯度的输入，用输出梯度和其他输入构造反向 `einsum`。
   - 对只出现在该输入中的被规约标签，插入单例维度并 broadcast 回原输入形状。
3. 增加测试：
   - 矩阵乘前向和梯度。
   - batch attention score 前向和梯度形状。
   - 单输入规约前向和梯度。
   - 隐式输出。
   - 不支持语义的报错。
   - 可选 CUDA。
4. 增加 PyTorch 兼容示例：
   - 同一文件通过 `USE_TORCH_1K=0/1` 切换实现。
5. 更新设计文档阶段记录。

## 非目标

1. 不实现 ellipsis：例如 `"...ij,jk->...ik"`。
2. 不实现同一输入内部重复标签的 diagonal / trace 语义：例如 `"ii->i"`。
3. 不实现 PyTorch 的 sublist 格式。
4. 不追求完整 dtype promotion。

## 验收标准

1. 新增 einsum 测试通过。
2. 新增 einsum 示例在 `torch_1k` 与 PyTorch 两条路径下通过。
3. PLAN-021、PLAN-022 新增测试仍通过。
4. 全量测试不回归。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
