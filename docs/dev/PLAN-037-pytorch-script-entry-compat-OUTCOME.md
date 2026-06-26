# PLAN-037 PyTorch 脚本入口兼容层结果

日期：2026-06-26

计划文件：`docs/dev/PLAN-037-pytorch-script-entry-compat.md`

实施基线 Git：`ba43885`

结果提交 Git：`待回填`

## 实现内容

1. 新增 `torch_1k.device`：
   - 支持 `torch.device("cpu")`。
   - 支持 `torch.device("cuda")` / `torch.device("cuda:0")` 的常见脚本写法。
   - 当前 CUDA index 仅作为脚本兼容信息保存，底层仍使用现有单 CUDA 后端。
2. 扩展 Tensor 入口接口：
   - `Tensor.to(device)`
   - `Tensor.to(dtype)`
   - `Tensor.to(device=..., dtype=...)`
   - `Tensor.to(other_tensor)`
   - `Tensor.dim()`
   - `Tensor.numel()`
   - `Tensor.is_cuda`
3. 扩展创建函数常见 dtype 参数：
   - `torch_1k.rand(..., dtype=...)`
   - `torch_1k.randn(..., dtype=...)`
   - `torch_1k.zeros(..., dtype=...)`
   - `torch_1k.ones(..., dtype=...)`
4. 扩展 Module 入口接口：
   - `Module.to(device=..., dtype=...)`
   - `Module.children()` / `named_children()`
   - `Module.modules()` / `named_modules()`
   - `Module.zero_grad(set_to_none=...)`
5. 扩展 Optimizer 入口接口：
   - `Optimizer.zero_grad(set_to_none=True)` 显式清为 `None`。
   - `Optimizer.zero_grad(set_to_none=False)` 保留填零行为。
6. 新增轻量 nn 容器：
   - `nn.Identity`
   - `nn.ModuleList`
7. 新增 `examples/example37_pytorch_script_entry_compat.py`：
   - 覆盖 `torch.device`、`to(device=..., dtype=...)`、`ModuleList`、`Identity`、模块遍历、`zero_grad(set_to_none=True)`。
   - 默认使用 `torch_1k`，`USE_TORCH_1K=0` 时切换到 PyTorch。
8. 新增 `tests/test_43_pytorch_script_entry_compat.py`：
   - 覆盖新增入口 API 的单元行为。
   - 运行 `torch_1k` 示例路径。
   - 子进程运行 PyTorch 替换路径。
9. 更新设计路线图，记录 PLAN-037 阶段更新。

## 验证结果

1. 默认 `torch_1k` 示例：
   - 命令：`python examples/example37_pytorch_script_entry_compat.py`
   - 结果：通过
   - accuracy：`1.000000`
2. PyTorch 导入替换路径：
   - 命令：`USE_TORCH_1K=0 python examples/example37_pytorch_script_entry_compat.py`
   - 结果：通过
   - accuracy：`1.000000`
3. PLAN-036 / PLAN-037 组合测试：
   - 命令：`pytest -q tests/test_42_pytorch_training_baseline.py tests/test_43_pytorch_script_entry_compat.py`
   - 结果：`6 passed`
4. 全量测试：
   - 命令：`pytest -q`
   - 结果：`248 passed`
5. 编译检查：
   - 命令：`python -m compileall -q torch_1k examples`
   - 结果：通过

## TODO / 未完成事项

1. `torch.device` 不实现完整多 GPU index 管理，`cuda:0` 仍归一到当前 CuPy CUDA 后端。
2. `Tensor.to(...)` 不实现 `non_blocking`、`copy`、`memory_format` 等高级参数。
3. `ModuleList` 不实现完整 list 变更语义，只覆盖 append、extend、索引、切片、迭代和长度。
4. 仍不实现完整 dtype promotion。
5. 仍不实现完整 PyTorch `nn.Module` hook / buffer / training API。

## 复核结论

本轮补齐的是脚本启动和模型装配阶段最常见的 PyTorch 入口层 API。PLAN-036 已证明核心训练链路可用；PLAN-037 进一步降低真实 PyTorch 脚本在进入核心训练前因为 `device`、`to`、模块容器或 `zero_grad` 签名差异失败的概率。实现仍保持教学型边界，没有扩展到完整 PyTorch dtype、hook 或多设备体系。
