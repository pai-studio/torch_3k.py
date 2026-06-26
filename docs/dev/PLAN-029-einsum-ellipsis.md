# PLAN-029 einsum ellipsis 常用路径

日期：2026-06-25

Git 基线：`81ffeb1`

说明：当前工作区已包含 PLAN-028 的未提交改动，本计划在该状态上继续实施。

## 背景

PLAN-023 和 PLAN-025 已实现并加固 `torch.einsum` 的核心子集，覆盖矩阵乘、batch matmul、attention、点积、规约和广播反向。但结果文档仍记录一个高价值缺口：不支持 ellipsis，例如 `"...ij,jk->...ik"`。

ellipsis 是真实模型代码中常见写法，尤其用于把任意 batch 维保留下来：

```python
torch.einsum("...ij,...jk->...ik", x, w)
torch.einsum("...qd,...kd->...qk", query, key)
torch.einsum("...ij->...", x)
```

当前用户必须把 batch 维写成固定标签，降低 PyTorch 替换兼容性，也限制了泛化 shape 的教学示例。

## 目标

1. 支持每个输入 spec 中最多一个 `...`。
2. 支持显式输出中的 `...`。
3. 支持隐式输出：
   - ellipsis 维度保留在输出最前面。
   - 只出现一次的普通标签按现有规则输出。
4. 支持 ellipsis 维度广播：
   - 多输入 ellipsis 维按右对齐广播。
   - operand 缺失的 ellipsis 维在反向传播中自动规约。
5. 保持现有非 ellipsis equation 行为不变。
6. 保持 CPU/CUDA 后端一致。

## 实施步骤

1. 改造 `torch_1k/functional/matrix.py`：
   - 将 equation 解析拆成模板解析与按输入 shape 展开。
   - 为 ellipsis 维分配不与用户标签冲突的内部标签。
   - 将 ellipsis equation 展开为 NumPy/CuPy 可执行的普通 `einsum` equation。
   - 复用现有 backward equation 构造与 `_restore_einsum_grad_shape(...)`。
2. 更新 `tests/test_32_einsum_api.py`：
   - 将 ellipsis 从 unsupported case 中移出。
   - 新增 `"...ij,jk->...ik"`、`"...ij,...jk->...ik"`、`"...ij->..."` 等 PyTorch 对比。
   - 覆盖 ellipsis broadcasting 和隐式输出。
   - 覆盖 CUDA 可选路径。
3. 新增示例：
   - 用 ellipsis 写泛化 batch matmul / attention。
   - 同进程对比 `torch_1k` 与 PyTorch 的前向和梯度。
4. 更新 PLAN-023 / PLAN-025 结果文档中的过期 TODO。
5. 更新设计路线图。

## 非目标

1. 不实现同一输入内部 repeated labels / diagonal，例如 `"ii->i"`。
2. 不实现 PyTorch sublist equation 格式。
3. 不实现完整 dtype promotion。
4. 不实现 `out=` 参数。

## 验收标准

1. 新增 ellipsis 测试通过。
2. 既有 `einsum` 测试不回归。
3. 新增示例与 PyTorch 对比通过。
4. 全量测试通过。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
