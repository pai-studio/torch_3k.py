# PLAN-002 核心定位与 CUDA 后端基础结果

日期：2026-06-24

## 完成内容

1. 新增 `docs/design/torch-3k-20260624-core-cuda-roadmap.md`，记录项目新定位、真实训练核心链路、核心模块分层和 CUDA/CuPy 支持策略。
2. 新增 PyTorch 兼容示例 `examples/example3_pytorch_compatible_train.py`。训练主体覆盖 `cuda.is_available()`、`.to(device)`、`Module.to(device)`、`train()`、`eval()`、`no_grad()`、`nn.Linear`、`nn.MSELoss`、`optim.SGD`，只需切换顶部导入即可在 PyTorch 与 `torch_1k` 间切换。
3. 新增 `torch_1k/backend.py` 与 `torch_1k/cuda.py`，提供 NumPy/CuPy 后端识别、设备迁移、CUDA 可用性检测。
4. 扩展 `Tensor` 支持 `device`、`to()`、`cpu()`、`cuda()`、`detach()`，并让 CUDA Tensor 的 `numpy()` 自动转回 NumPy。
5. 扩展顶层兼容接口：`torch_1k.tensor`、`zeros`、`ones`、`cuda.is_available()`。
6. 改造基础算子，使 `exp`、`sin`、`cos`、`tanh`、`sigmoid`、`transpose`、`broadcast_to`、`sum`、`pad` 等根据输入数组使用 NumPy 或 CuPy。
7. 修复稳定性问题：
   - `nn.functional.relu` 缺少冒号。
   - `Sigmoid.backward` 使用未定义变量。
   - `Module.zero_grad` 写成 `self.self.parameters()`。
   - `Tensor.randn` 返回全零。
   - `DataLoader` 缺少 `__iter__`。
   - `Sub`、`Mul`、`Div` 缺少广播梯度还原。
8. 新增 `tests/test_11_pytorch_compat.py`，覆盖 PyTorch 风格设备接口、标量广播梯度、可选 CUDA Tensor 反传。
9. 新增 `pytest.ini`，保证从仓库根目录运行 pytest 时能找到本地包。

## 验证结果

已运行：

```bash
python examples/example3_pytorch_compatible_train.py
USE_TORCH_1K=0 python examples/example3_pytorch_compatible_train.py
pytest -q tests/test_11_pytorch_compat.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- `torch_1k` 示例：`final_loss=1.008334`
- PyTorch 示例：`final_loss=1.049498`
- 新增测试：`3 passed`
- 全量测试：`37 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

本机 `torch_1k.cuda.is_available()` 返回 `False`，因此 CUDA 实算路径由可选测试保护；有 CuPy/CUDA 环境时会执行 CUDA Tensor 的基础计算与反传断言。

## 代码规模

当前 `torch_1k` 核心代码统计约 `1240` 行，仍低于新定位建议的 3000 行以内。
