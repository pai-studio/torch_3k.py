# PLAN-031 nonzero 与 where(condition) 坐标 API

日期：2026-06-25

Git 基线：`81ffeb1`

说明：当前工作区已包含 PLAN-028 至 PLAN-030 的未提交改动，本计划在该状态上继续实施。

## 背景

PLAN-027 已补齐比较掩码、`where(condition, input, other)`、`masked_fill`、`index_select`、`gather` 和数值工具；PLAN-030 又公开了 `scatter` / `scatter_add`。但 PLAN-027 结果文档仍记录一个常用缺口：尚未实现 `where(condition)` 单参数形式。

在 PyTorch 中：

```python
torch.where(condition)
```

等价于：

```python
torch.nonzero(condition, as_tuple=True)
```

它常用于从 mask 中取坐标、构造稀疏索引、调试有效位置，属于 mask/index 胶水 API 的自然补全。

## 目标

1. 新增 `torch_1k.nonzero(input, as_tuple=False)`：
   - 默认返回二维坐标 Tensor，shape 为 `(z, input.ndim)`。
   - `as_tuple=True` 返回每个维度一个 1D index Tensor。
   - 返回 Tensor 均不可微。
2. 扩展 `torch_1k.where(...)`：
   - 保留三参数 `where(condition, input, other)` 的可微分流语义。
   - 新增单参数 `where(condition)`，返回 `nonzero(condition, as_tuple=True)`。
   - 如果只传 `input` 或只传 `other`，明确报错。
3. 新增 Tensor 方法：
   - `Tensor.nonzero(as_tuple=False)`。
4. 支持 CPU/CUDA 后端一致。
5. 与 PyTorch 对比常见 bool mask 和 numeric input。

## 实施步骤

1. 改造 `torch_1k/functional/numeric.py`：
   - 新增 `nonzero(...)` 函数。
   - 扩展 `where(...)` 函数签名。
2. 改造 `torch_1k/tensor.py`：
   - 增加 `nonzero(...)` 方法。
3. 新增测试：
   - `nonzero(as_tuple=False)` 与 PyTorch 对比。
   - `nonzero(as_tuple=True)` 与 PyTorch 对比。
   - `where(condition)` 与 PyTorch 对比。
   - 三参数 `where` 旧路径回归。
   - numeric input 非零坐标。
   - CUDA 可选路径。
4. 新增示例：
   - 从 mask 取有效坐标。
   - 用 `where(condition)` 得到 tuple index。
   - 同进程对比 `torch_1k` 与 PyTorch。
5. 更新 PLAN-027 结果文档和设计路线图。

## 非目标

1. 不实现完整高级索引语义。
2. 不实现 `out=` 参数。
3. 不实现完整 dtype promotion。

## 验收标准

1. 新增 nonzero / where(condition) 测试通过。
2. 既有 mask/index/scatter 测试不回归。
3. 新增示例与 PyTorch 对比通过。
4. 全量测试通过。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
