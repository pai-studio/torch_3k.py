# PLAN-020 常用张量创建 API

日期：2026-06-24

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 目标

补齐真实训练和模型代码中高频使用的张量创建 API，使 `torch.arange`、`torch.zeros_like`、`torch.ones_like`、`torch.full`、`torch.full_like` 和 PyTorch 风格 `torch.randint` 可直接运行，并保持 CPU/CUDA 后端一致。

本计划聚焦常用创建函数，不实现完整 dtype promotion 体系。

## 当前缺口

1. 顶层缺少 `torch.arange`，但位置索引、标签构造、mask 构造经常需要它。
2. `Tensor` 只有类方法 `zeros_like` / `ones_like`，顶层未导出 `torch.zeros_like` / `torch.ones_like`。
3. 缺少 `torch.full` / `torch.full_like`。
4. 当前 `torch.randint` 签名偏 NumPy，不能按常见 PyTorch 写法 `torch.randint(low, high, size)` 使用。
5. 创建函数需要支持 `device='cuda'`，like 函数需要默认继承输入 Tensor 的 device。

## 实施步骤

1. 在 `torch_1k.tensor` 中新增或修正顶层创建函数：
   - `arange(start, end=None, step=1, device=None, dtype=None, requires_grad=False)`
   - `zeros_like(input, device=None, dtype=None, requires_grad=False)`
   - `ones_like(input, device=None, dtype=None, requires_grad=False)`
   - `full(size, fill_value, device=None, dtype=None, requires_grad=False)`
   - `full_like(input, fill_value, device=None, dtype=None, requires_grad=False)`
   - `randint(low, high=None, size=None, device=None, dtype=None, requires_grad=False)`
2. 保留 `Tensor.zeros_like` / `Tensor.ones_like` 可用。
3. 更新 `torch_1k.__init__` 导出新创建函数。
4. 新增 `examples/example20_tensor_creation_compare.py`，覆盖 PyTorch import 切换路径。
5. 新增 `tests/test_29_tensor_creation_api.py`，覆盖：
   - `arange` 单参数/双参数/step。
   - `zeros_like` / `ones_like` 继承 shape/device。
   - `full` / `full_like`。
   - `randint(low, high, size)`。
   - 默认不建图。
   - 可选 CUDA 路径。
6. 更新设计文档记录 PLAN-020。
7. 运行新增测试、示例、全量测试和编译检查。

## 验收标准

1. `torch.arange(5)` 返回 `[0, 1, 2, 3, 4]`。
2. `torch.arange(1, 6, 2)` 返回 `[1, 3, 5]`。
3. `torch.zeros_like(x)` / `torch.ones_like(x)` 默认继承 `x` 的 shape 和 device。
4. `torch.full((2, 3), 7)` / `torch.full_like(x, 7)` 可用。
5. `torch.randint(0, 10, (2, 3))` 可用。
6. CPU 全量测试通过；CUDA 可用时新增创建函数保持 CUDA 设备。

## TODO 记录

1. 本计划不实现 `empty`，避免暴露未初始化内存这一教学场景中价值较低的行为。
2. 本计划不实现完整 dtype promotion。
