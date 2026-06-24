# PLAN-018 cat / concat API 兼容结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-018-cat-api.md`

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 实施结果

1. 在 `functional.matrix` 中新增可反向传播的 `Cat`：
   - 支持非空 Tensor 序列。
   - 支持 `dim` / `axis`。
   - 支持负维度。
   - 校验除拼接维度外的 shape 一致。
   - 前向使用 NumPy/CuPy `concatenate`。
   - 反向沿拼接维度切片，把梯度分回每个输入。
2. 新增顶层 API：
   - `torch.cat(tensors, dim=0)`
   - `torch.concat(tensors, dim=0)`
   - `torch.concatenate(tensors, dim=0)`
3. 新增 `examples/example18_cat_api_compare.py`：
   - 覆盖多分支特征拼接。
   - 覆盖 `cat` / `concat`。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
4. 更新设计文档，记录 `cat` / `concat` 已补齐。

## 新增测试

新增 `tests/test_27_cat_api.py`，覆盖：

1. `torch.cat([x1, x2], dim=0)` 前向和反向。
2. `torch.cat(..., dim=-1)` 前向和反向。
3. `torch.concat` / `torch.concatenate` 别名。
4. 普通 Python 输入与 Tensor 混合拼接。
5. 空输入和 shape 不匹配报错。
6. 可选 CUDA Tensor 拼接、输出设备和梯度设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_27_cat_api.py
python examples/example18_cat_api_compare.py
USE_TORCH_1K=0 python examples/example18_cat_api_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 cat API 测试：`6 passed`
- cat API 双后端示例：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`118 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 `split`、`chunk`、`repeat`。
2. 尚未实现 PyTorch 对所有序列类型和 dtype promotion 的完整语义。

## 复核结论

本轮补齐了真实模型 forward 中常见的特征拼接能力。`torch.cat` / `torch.concat` 现在可用于多分支网络和跳连结构，并且梯度能按拼接维度正确切片回传到每个输入。
