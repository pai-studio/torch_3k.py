# PLAN-003 PyTorch 兼容 MLP 训练示例结果

日期：2026-06-24

## 完成内容

1. 新增 `examples/example4_mlp_train_compare.py`，使用同一份训练主体跑 PyTorch 或 `torch_1k`：
   - 默认 `torch_1k` 路径。
   - 设置 `USE_TORCH_1K=0` 后跑 PyTorch 路径。
2. 示例实现 XOR 分类 MLP：
   - `nn.Sequential`
   - `nn.Linear`
   - `nn.ReLU`
   - `nn.CrossEntropyLoss`
   - `optim.SGD`
   - `torch.argmax`
3. 新增 `nn.Sequential` 和 `nn.ReLU` 模块。
4. 新增 `nn.CrossEntropyLoss`，内部使用稳定 softmax 交叉熵公式。
5. 扩展 Tensor 与顶层接口：
   - `Tensor.mean()`
   - `Tensor.float()`
   - `Tensor.long()`
   - `Tensor.__eq__`
   - `torch_1k.argmax`
   - `torch_1k.float32`、`torch_1k.float64`、`torch_1k.long`
6. 修正 `Tensor.backward()` 初始化输出梯度时的后端选择，使 CUDA Tensor 也能创建同设备梯度。
7. 新增 `tests/test_12_mlp.py`，覆盖交叉熵反传形状和 XOR MLP 训练收敛。

## 验证结果

已运行：

```bash
python examples/example4_mlp_train_compare.py
USE_TORCH_1K=0 python examples/example4_mlp_train_compare.py
pytest -q tests/test_12_mlp.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- `torch_1k` MLP 示例：`final_loss=0.005656`，`accuracy=1.000000`
- PyTorch MLP 示例：`final_loss=0.003597`，`accuracy=1.000000`
- 新增测试：`2 passed`
- 全量测试：`39 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## 代码规模

当前 `torch_1k` 核心代码统计约 `1314` 行，仍低于新定位建议的 3000 行以内。
