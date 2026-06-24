# PLAN-025 einsum 经典用法与广播反向加固

日期：2026-06-25

Git 基线：`a5d4d3b`

说明：当前工作区已包含 PLAN-021 至 PLAN-024 的未提交改动，本计划在该状态上继续实施。

## 背景

`einsum` 是深度学习代码中的高价值核心算子。PLAN-023 已实现基础子集，但 review 发现一个关键问题：当前反向传播对前向广播维度处理不正确。

复现：

```python
x = torch_1k.tensor([[1.0], [2.0]], requires_grad=True)
y = torch_1k.tensor([3.0, 4.0, 5.0], requires_grad=True)
z = torch_1k.einsum("ij,j->ij", x, y)
z.sum().backward()
```

PyTorch 中 `x.grad` 应为 `[[12.0], [12.0]]`，当前实现会尝试把 `(2, 3)` 梯度直接 reshape 为 `(2, 1)` 而报错。广播反向必须对广播出来的维度求和，而不是 reshape。

## 经典用法范围

本轮以 PyTorch 常见写法作为对比基准，覆盖：

1. 矩阵乘：`"ij,jk->ik"`。
2. 批量矩阵乘：`"bij,bjk->bik"`。
3. 点积：`"i,i->"`。
4. 外积：`"i,j->ij"`。
5. 单输入规约：`"ij->i"`、`"ij->"`。
6. 逐元素乘加：`"ij,ij->"`。
7. attention score：`"bqd,bkd->bqk"`。
8. attention context：`"bqk,bkd->bqd"`。
9. 广播逐元素乘：`"ij,j->ij"`。
10. 广播乘后规约：`"ij,j->i"`。

## 目标

1. 修复 `Einsum.backward` 的广播维度还原。
2. 对核心支持子集增加 PyTorch 对比测试：
   - 前向数值一致。
   - 所有输入梯度一致。
3. 新增更详细的 `einsum` 使用示例，展示常见写法和结果。
4. 保持 PLAN-023 明确的非目标不变：
   - 不实现 ellipsis。
   - 不实现同一输入内部 repeated labels / diagonal。
   - 不实现 sublist 格式。

## 实施步骤

1. 在 `torch_1k/functional/matrix.py` 中新增 einsum 梯度对齐辅助函数：
   - 先按输入 spec 补齐缺失 label 的单例维。
   - 对前向广播维求和还原。
   - 对私有规约维 broadcast 回输入形状。
2. 更新 `tests/test_32_einsum_api.py`：
   - 增加广播反向回归测试。
   - 增加 PyTorch 经典用法参数化对比测试。
3. 新增 `examples/example25_einsum_usage_compare.py`：
   - 使用同一代码在 `torch_1k` 和 PyTorch 两条路径运行。
   - 覆盖矩阵乘、batch matmul、attention score/context、广播门控、规约。
4. 更新 PLAN-023 结果文档，记录本轮 hardening 关联。
5. 更新设计文档阶段记录。

## 验收标准

1. 新增 einsum hardening 测试通过。
2. 新增详细示例在 `torch_1k` 与 PyTorch 两条路径下通过。
3. PLAN-021 至 PLAN-025 新增测试集合通过。
4. 全量测试不回归。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录修复点、对比范围和验证结果。
