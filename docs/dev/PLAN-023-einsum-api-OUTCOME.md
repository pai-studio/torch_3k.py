# PLAN-023 einsum 核心算子结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-023-einsum-api.md`

Git 基线：`a5d4d3b`

说明：本轮在包含 PLAN-021 和 PLAN-022 未提交改动的工作区状态上继续实施。

## 实施结果

1. 新增顶层 `torch_1k.einsum(equation, *operands)`。
2. 支持显式输出 equation：
   - 例如 `"ij,jk->ik"`。
   - 例如 `"bqd,bkd->bqk"`。
3. 支持隐式输出 equation：
   - 例如 `"ij,jk"`。
4. 支持输出为标量的规约：
   - 例如 `"i,i->"`。
5. 支持单输入规约：
   - 例如 `"ij->i"`。
   - 反向传播会把缺失的被规约维度 broadcast 回输入形状。
6. 反向传播使用统一策略：
   - 对每个需要梯度的输入，用输出梯度和其他输入构造反向 `einsum`。
   - 对仅存在于该输入中的被规约标签，插入单例维度并广播回原输入形状。
7. CPU/CUDA 后端一致：
   - 前向使用输入所在后端的 `einsum`。
   - 反向梯度也保持同一后端。
8. 新增 `examples/example23_einsum_compare.py`：
   - 用 `einsum` 表达 attention score 和 context。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
9. 更新设计文档，记录 PLAN-023 已补齐 `einsum` 核心子集。

## 新增测试

新增 `tests/test_32_einsum_api.py`，覆盖：

1. `"ij,jk->ik"` 的矩阵乘前向和反向梯度。
2. `"ij,jk"` 隐式输出与显式矩阵乘一致。
3. `"bqd,bkd->bqk"` attention score 前向形状和反向梯度形状。
4. `"ij->i"` 单输入规约前向和梯度 broadcast。
5. `"i,i->"` 点积标量输出和双输入梯度。
6. 不支持的 repeated labels 和 ellipsis 报错。
7. 可选 CUDA 路径下输出和梯度保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_32_einsum_api.py
python examples/example23_einsum_compare.py
USE_TORCH_1K=0 python examples/example23_einsum_compare.py
pytest -q tests/test_30_loss_functional_reduction.py tests/test_31_shape_split_chunk_repeat.py tests/test_32_einsum_api.py
python -m compileall -q torch_1k examples
pytest -q
python examples/example21_loss_functional_reduction_compare.py
USE_TORCH_1K=0 python examples/example21_loss_functional_reduction_compare.py
python examples/example22_shape_split_chunk_repeat_compare.py
USE_TORCH_1K=0 python examples/example22_shape_split_chunk_repeat_compare.py
```

结果：

- 新增 einsum 测试：`7 passed`
- PLAN-021/022/023 新增测试集合：`19 passed`
- 新增示例 21、22、23：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`149 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. ellipsis 已在 PLAN-029 补齐，例如 `"...ij,jk->...ik"`。
2. 同一输入内部 repeated labels 的 diagonal / trace 语义已在 PLAN-032 补齐，例如 `"ii->i"`。
3. 尚未实现 PyTorch 的 sublist equation 格式。
4. 尚未实现完整 dtype promotion。

## 后续加固

PLAN-025 已修复 PLAN-023 初始实现中的广播反向问题，并补充 PyTorch 经典用法前向与梯度对比，包括矩阵乘、batch matmul、attention、外积、点积、规约和广播乘法。

## 复核结论

本轮补齐了一个高价值核心张量算子。`einsum` 现在能覆盖矩阵乘、批量 attention 打分、context 聚合、点积和普通规约，同时反向传播实现保持在教学代码可读范围内。
