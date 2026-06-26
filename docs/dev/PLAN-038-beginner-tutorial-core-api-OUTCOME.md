# PLAN-038 新手教程驱动基础 API 加固结果

日期：2026-06-26

计划文件：`docs/dev/PLAN-038-beginner-tutorial-core-api.md`

设计文档：`docs/design/torch-tutorial-20260626-api-dependency.md`

实施基线 Git：`6b1c9bb3c61fb9657a4f3a6c1546e24eee79e2fc`

结果提交 Git：`待回填`

## 实现内容

1. 新增设计文档：
   - 从 torch 新手教程场景反推最终模型、依赖函数、底层必须能力和 wrapper 边界。
   - 明确教程路径：张量基础、自动微分、手写线性回归、`nn.Module` MLP、DataLoader mini-batch、CNN/Transformer 方向感。
2. 新增教程式例子 `examples/example38_beginner_tutorial.py`：
   - 默认使用 `torch_1k`。
   - `USE_TORCH_1K=0` 时切换到 PyTorch。
   - 覆盖张量基础、NumPy 转换、自动微分、手写线性回归、MLP 分类和 DataLoader mini-batch。
3. 新增测试 `tests/test_44_beginner_tutorial_core_api.py`：
   - 与 PyTorch 对比基础 wrapper 的前向和关键梯度。
   - 验证 `nn.Tanh` / `nn.Sigmoid`。
   - 运行教程例子的 `torch_1k` 和 PyTorch 两条路径。
4. 补齐底层和基础 wrapper：
   - `torch_1k.eye`
   - `torch_1k.as_tensor`
   - `torch_1k.from_numpy`
   - `torch_1k.rand_like`
   - `torch_1k.randn_like`
   - 可微 `Tensor.clone()`
   - `Tensor.tolist()`
   - `Tensor.matmul()` / `Tensor.mm()`
   - `Tensor.__len__`
   - `torch_1k.sum(input, dim=..., keepdim=...)`
   - `torch_1k.mean(input, dim=..., keepdim=...)`
   - `Tensor.sum(..., keepdim=...)`
   - `Tensor.mean(..., keepdim=...)`
   - `nn.Tanh`
   - `nn.Sigmoid`
5. 扩展便利创建函数：
   - `torch_1k.linspace(..., dtype=...)`
   - `torch_1k.normal(..., dtype=...)`
6. 更新设计路线图，记录 PLAN-038 阶段更新。

## 验证结果

1. 默认 `torch_1k` 教程例子：
   - 命令：`python examples/example38_beginner_tutorial.py`
   - 结果：通过
   - 手写回归 loss：`0.000013`
   - MLP accuracy：`1.000000`
2. PyTorch 导入替换路径：
   - 命令：`USE_TORCH_1K=0 python examples/example38_beginner_tutorial.py`
   - 结果：通过
   - 手写回归 loss：`0.000030`
   - MLP accuracy：`1.000000`
3. 新增测试：
   - 命令：`pytest -q tests/test_44_beginner_tutorial_core_api.py`
   - 结果：`5 passed`
4. PLAN-036 / PLAN-037 / PLAN-038 组合测试：
   - 命令：`pytest -q tests/test_42_pytorch_training_baseline.py tests/test_43_pytorch_script_entry_compat.py tests/test_44_beginner_tutorial_core_api.py`
   - 结果：`11 passed`
5. 全量测试：
   - 命令：`pytest -q`
   - 结果：`253 passed`
6. 编译检查：
   - 命令：`python -m compileall -q torch_1k examples`
   - 结果：通过

## TODO / 未完成事项

1. 仍不实现完整 dtype promotion。
2. 仍不实现原地操作族。
3. 仍不实现完整高级索引。
4. 仍不实现 torchvision 风格数据集和 transform。
5. `from_numpy` / `as_tensor` 只覆盖当前 NumPy/CuPy 后端教学路径，不追完整 PyTorch storage alias 语义。

## 复核结论

本轮按“给 torch 新手写教程”倒推基础能力，而不是继续补单个高级 API。新增的底层可微 `clone` 和一组基础 wrapper 让入门教程可以自然表达张量创建、形状、规约、矩阵乘法、NumPy 转换、自动微分、手写回归、MLP 和 DataLoader。PLAN-036 / PLAN-037 的训练基线与脚本入口层均未回归。
