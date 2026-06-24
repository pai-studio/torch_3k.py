# PLAN-020 常用张量创建 API 结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-020-tensor-creation-api.md`

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 实施结果

1. 新增顶层 `torch.arange(...)`：
   - 支持 `arange(end)`。
   - 支持 `arange(start, end, step)`。
   - 支持 `device` / `dtype`。
2. 新增顶层 like 创建函数：
   - `torch.zeros_like(input, device=None, dtype=None)`
   - `torch.ones_like(input, device=None, dtype=None)`
   - 默认继承输入 shape、dtype 和 device。
3. 新增填充值创建函数：
   - `torch.full(size, fill_value, device=None, dtype=None)`
   - `torch.full_like(input, fill_value, device=None, dtype=None)`
4. 修正 `torch.randint(...)` 的常用 PyTorch 签名：
   - `torch.randint(high, size)`
   - `torch.randint(low, high, size)`
   - 默认 dtype 为 `int64`。
5. 更新 `torch_1k.__init__`，导出新增创建函数。
6. 新增 `examples/example20_tensor_creation_compare.py`：
   - 覆盖 `arange`、`randint`、`full`、`zeros_like`、`ones_like`、`full_like`。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
7. 更新设计文档，将常用创建函数纳入基础算子层。

## 新增测试

新增 `tests/test_29_tensor_creation_api.py`，覆盖：

1. `torch.arange(end)`。
2. `torch.arange(start, end, step)`。
3. `torch.zeros_like` / `torch.ones_like` 的 shape 和 dtype 继承。
4. `torch.full` / `torch.full_like`。
5. `torch.randint(high, size)` 与 `torch.randint(low, high, size)`。
6. 默认不追踪梯度。
7. 可选 CUDA 路径下输出保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_29_tensor_creation_api.py
python examples/example20_tensor_creation_compare.py
USE_TORCH_1K=0 python examples/example20_tensor_creation_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增张量创建 API 测试：`5 passed`
- 张量创建 API 双后端示例：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`130 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 `empty`，避免在教学代码中暴露未初始化内存行为。
2. 尚未实现完整 dtype promotion。

## 复核结论

本轮补齐了真实训练和模型代码中常见的张量创建接口。现在位置索引、mask 构造、like Tensor 初始化和随机整数标签构造都可以按 PyTorch 风格直接写，并保持 CPU/CUDA 后端一致。
