# PLAN-015 常用 shape API 兼容结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-015-shape-api.md`

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 实施结果

1. 在 `functional.matrix` 中新增常用 shape API：
   - `view(x, shape)`
   - `unsqueeze(x, dim)`
   - `squeeze(x, dim=None, axis=None)`
   - `flatten(x, start_dim=0, end_dim=-1)`
2. 上述 API 均复用已有 `reshape`，因此反向传播会通过 `Reshape.backward` 回到原始 shape。
3. 给 `Tensor` 增加方法：
   - `x.view(...)`
   - `x.unsqueeze(dim)`
   - `x.squeeze(dim=None)`
   - `x.flatten(start_dim=0, end_dim=-1)`
4. 顶层 `torch.unsqueeze` 已改为复用 `functional.unsqueeze`，避免重复 shape 逻辑。
5. 顶层 `torch.squeeze` 和 `torch.flatten` 通过 `functional` 导出可用。
6. `nn.Flatten.forward` 已改为复用 `x.flatten(...)`。
7. 删除 `functional.matrix` 中未完成的 `_Unsqueeze` 占位实现。
8. 新增 `examples/example15_shape_api_compare.py`：
   - 覆盖 `squeeze`、`unsqueeze`、`flatten`、`view` 和 `nn.Flatten`。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
9. 更新设计文档，记录 PLAN-015 已补齐常用 shape API。

## 新增测试

新增 `tests/test_24_shape_api.py`，覆盖：

1. `x.view(3, -1)` 前向 shape 与反向传播。
2. `x.unsqueeze(dim)` 与 `torch.unsqueeze(x, dim)`。
3. `x.squeeze()`、`x.squeeze(dim=-2)` 和指定非单例维度时保持 shape。
4. `torch.squeeze(x, axis=(...))`。
5. `x.flatten(start_dim=1)`、`torch.flatten(x, start_dim=0, end_dim=1)` 和 `nn.Flatten`。
6. 非法维度与非法 flatten 区间报错。
7. 可选 CUDA Tensor 经 shape API 后仍保持 CUDA 设备，梯度也保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_24_shape_api.py
python examples/example15_shape_api_compare.py
USE_TORCH_1K=0 python examples/example15_shape_api_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 shape API 测试：`7 passed`
- shape API 双后端示例：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`97 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 `cat`、`split`、`chunk`、`repeat` 等更大的 shape API 面。
2. 尚未实现 PyTorch 的共享存储 view 语义；当前仍是教学型 Tensor 包装。
3. 高级索引仍不属于本轮范围。

## 复核结论

本轮补齐了真实模型 forward 中最常见的一组 shape 写法，尤其是 `x.view(batch, -1)`、`x.flatten(start_dim=1)` 和 `x.squeeze()/unsqueeze()`。这些能力直接复用已有自动微分实现，保持代码短小，同时提升 PyTorch 示例替换兼容性。
