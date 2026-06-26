# PLAN-037 PyTorch 脚本入口兼容层

日期：2026-06-26

Git 基线：`ba43885`

## 背景

PLAN-036 已建立训练脚本替换兼容基线，证明当前核心训练链路可以用同一训练主体在 `torch_1k` 与 PyTorch 下运行。下一类高价值缺口不是单个数学函数，而是真实 PyTorch 脚本启动阶段常见的入口层写法：

1. `torch.device(...)`。
2. `tensor.to(device=..., dtype=...)` 和 `model.to(device=..., dtype=...)`。
3. `optimizer.zero_grad(set_to_none=True)`。
4. `Tensor.dim()` / `Tensor.numel()` / `Tensor.is_cuda`。
5. `Module.children()` / `modules()` 等模块遍历接口。
6. `nn.ModuleList` / `nn.Identity` 等轻量容器。

这些接口不直接增加模型表达能力，但会显著减少“真实 PyTorch 代码刚启动就因入口 API 缺失失败”的情况。

## 目标

1. 新增 `torch_1k.device`，支持常见 `torch.device("cuda" if ... else "cpu")` 写法。
2. 扩展 Tensor / Module 迁移接口：
   - `Tensor.to(device)`
   - `Tensor.to(dtype)`
   - `Tensor.to(device=device, dtype=dtype)`
   - `Tensor.to(other_tensor)`
   - `Module.to(device=..., dtype=...)`
3. 补齐 Tensor 常见属性/方法：
   - `Tensor.dim()`
   - `Tensor.numel()`
   - `Tensor.is_cuda`
4. 补齐 Module 遍历接口：
   - `children()` / `named_children()`
   - `modules()` / `named_modules()`
5. 新增轻量 nn 容器：
   - `nn.Identity`
   - `nn.ModuleList`
6. 扩展 `Optimizer.zero_grad(set_to_none=...)` 与 `Module.zero_grad(set_to_none=...)`。
7. 新增 PyTorch 对照示例和测试，验证同一脚本只切换 import 即可运行。

## 实施步骤

1. 新增 `examples/example37_pytorch_script_entry_compat.py`：
   - 使用 `torch.device`、`Tensor.to(device=..., dtype=...)`、`Module.to(...)`。
   - 使用 `ModuleList` 和 `Identity` 构建模型。
   - 使用 `children()` / `modules()` 统计模块。
   - 使用 `optimizer.zero_grad(set_to_none=True)`。
   - 同一文件支持 `USE_TORCH_1K=0` 切换到 PyTorch。
2. 新增 `tests/test_43_pytorch_script_entry_compat.py`：
   - 验证新增入口 API。
   - 运行默认 `torch_1k` 示例。
   - 子进程运行 PyTorch 替换路径；无 PyTorch 时跳过。
3. 改造实现：
   - `backend.py` 增加 `device` 类型。
   - `tensor.py` 扩展 `to`、创建函数 dtype 参数和 Tensor 元信息方法。
   - `nn/module.py` 扩展 `to`、`zero_grad` 和模块遍历。
   - `nn/sequential.py` 增加 `Identity` / `ModuleList`。
   - `optim/optimizer.py` 增加 `set_to_none` 参数。
4. 更新路线图和结果文档。

## 非目标

1. 不实现完整 dtype promotion。
2. 不实现 `torch.device` 的完整 index / multi-GPU 管理能力；`cuda:0` 归一到当前 CUDA 后端。
3. 不实现 `ModuleList` 的完整 Python list 所有变更语义，只覆盖 append、extend、索引、迭代、len 和 forward 构图常用路径。
4. 不实现 `Tensor.to(memory_format=...)`、`non_blocking` 等高级参数。

## 验收标准

1. 新增入口兼容示例在 `torch_1k` 路径通过。
2. 新增入口兼容示例在 PyTorch 路径通过，或未安装 PyTorch 时测试明确跳过。
3. 新增测试通过。
4. PLAN-036 训练基线不回归。
5. 全量测试通过。
6. `python -m compileall -q torch_1k examples` 通过。
7. OUTCOME 记录实现范围、验证结果、结果 Git 记录和未完成事项。
