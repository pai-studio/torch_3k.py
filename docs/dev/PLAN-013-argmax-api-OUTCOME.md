# PLAN-013 argmax API 兼容结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-013-argmax-api.md`

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 实施结果

1. 新增设备感知的非可微 `argmax` 实现：
   - 支持 `dim` / `axis`。
   - 支持 `keepdim` / `keepdims`。
   - 返回 int64 Tensor。
   - 返回结果始终 `requires_grad=False`。
2. 新增 `Tensor.argmax(dim=None, keepdim=False, axis=None, keepdims=None)`：
   - 支持 `logits.argmax(dim=1)` 分类推理写法。
   - 支持 `keepdim=True` 保留规约维度。
3. 更新顶层 `torch.argmax(...)`：
   - 复用 functional 层实现。
   - 保持 `torch.argmax(x, dim=...)` 兼容。
   - 新增 `keepdim` 兼容。
4. 新增 `examples/example13_argmax_api_compare.py`：
   - 覆盖 Tensor 方法形式。
   - 覆盖顶层函数形式。
   - 覆盖 `keepdim=True`。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
5. 更新设计文档：
   - 将基础算子中的规约条目扩展为“规约与预测索引”。
   - 记录 PLAN-013 已补齐分类推理常用 `argmax` API。

## 新增测试

新增 `tests/test_22_argmax_api.py`，覆盖：

1. 全局 `torch.argmax(x)` 返回扁平化最大值位置。
2. `torch.argmax(x, dim=1)`。
3. `x.argmax(dim=1)`。
4. `x.argmax(dim=1, keepdim=True)`。
5. `axis` / `keepdims` 别名。
6. 非 Tensor 输入。
7. tuple dim 明确报错。
8. 可选 CUDA 输入保持 CUDA 输出。

## 验证结果

已运行：

```bash
pytest -q tests/test_22_argmax_api.py
python examples/example13_argmax_api_compare.py
USE_TORCH_1K=0 python examples/example13_argmax_api_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 argmax API 测试：`7 passed`
- argmax API 双后端示例：`torch_1k` 与 PyTorch 输出一致
- 全量测试：`82 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. `topk` 已由 PLAN-024 补齐，`sort` / `argsort` 已由 PLAN-026 补齐。
2. 尚未实现 `argmax` 的 tuple 维度语义。

## 复核结论

本轮补齐了分类训练和推理中高频使用的 `argmax` 兼容接口。现在示例和用户代码可以使用更常见的 `logits.argmax(dim=1)` 写法，同时保持结果非可微和 CPU/CUDA 后端一致。
