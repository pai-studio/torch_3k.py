# PLAN-018 cat / concat API 兼容

日期：2026-06-24

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 目标

补齐真实模型 forward 中高频使用的张量拼接 API，使 `torch.cat([...], dim=...)`、`torch.concat([...], dim=...)` 和 `torch.concatenate([...], dim=...)` 可直接运行，并支持反向传播和 CUDA/CuPy 后端。

本计划聚焦拼接，不扩展到 split/chunk/repeat 等更大的 shape API。

## 当前缺口

1. PLAN-015 结果中记录仍未实现 `cat`、`split`、`chunk`、`repeat` 等更大的 shape API。
2. 多分支模型、特征融合、跳连结构中常用 `torch.cat` 拼接张量。
3. 当前 `torch_1k` 只有 `stack`，但 `stack` 会新增维度，不能替代 `cat`。
4. 缺少拼接操作的反向传播切片回填。

## 实施步骤

1. 在 `functional.matrix` 中新增 `Cat(Function)`：
   - 支持非空 Tensor 序列。
   - 支持 `dim` / `axis` 参数。
   - 支持负维度。
   - 校验除拼接维度外的 shape 一致。
   - 前向使用 `xp.concatenate`。
   - 反向沿拼接维度切片，把 `gy` 分回每个输入。
2. 新增顶层函数：
   - `torch.cat(tensors, dim=0)`
   - `torch.concat(tensors, dim=0)`
   - `torch.concatenate(tensors, dim=0)`
3. 新增 `examples/example18_cat_api_compare.py`，覆盖 PyTorch import 切换路径。
4. 新增 `tests/test_27_cat_api.py`，覆盖：
   - dim=0 拼接与反向传播。
   - dim=1 / 负维度拼接。
   - `concat` / `concatenate` 别名。
   - 非 Tensor 输入。
   - shape 不匹配报错。
   - 可选 CUDA 输入。
5. 更新设计文档记录 PLAN-018。
6. 运行新增测试、示例、全量测试和编译检查。

## 验收标准

1. `torch.cat([x1, x2], dim=0)` 前向结果正确。
2. `torch.cat([x1, x2], dim=1).sum().backward()` 能把梯度分回每个输入。
3. `torch.concat` 和 `torch.concatenate` 与 `torch.cat` 行为一致。
4. CUDA Tensor 拼接后仍保持 CUDA 设备。
5. 全量测试通过。

## TODO 记录

1. 本计划不实现 `split`、`chunk`、`repeat`。
2. 本计划不实现 PyTorch 对所有序列类型和高级 dtype promotion 的完整语义。
