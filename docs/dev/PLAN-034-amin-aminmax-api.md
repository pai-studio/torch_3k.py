# PLAN-034 amin / aminmax 规约 API

日期：2026-06-26

Git 基线：`c125f9d`

## 背景

PLAN-033 已补齐 `torch.amax()` 和 `Tensor.amax()`，支持全局、单维、多维 tuple 规约、`keepdim` 和重复最大值梯度平均分配。但结果文档仍记录一个直接相邻的高价值缺口：

1. 尚未实现 `torch.amin`。
2. 尚未实现 `torch.aminmax`。

真实训练和评估代码中，最小值规约常用于数值范围检查、mask 后的边界统计、归一化前后的稳定性监控；`aminmax` 则常用于一次性提取张量区间。补齐这组 API 可以让 `amax` 之后的规约能力更完整，也保持 PyTorch 替换路径一致。

## 目标

1. 新增顶层函数：
   - `torch_1k.amin(input, dim=None, keepdim=False)`
   - `torch_1k.aminmax(input, dim=None, keepdim=False)`
2. 新增 Tensor 方法：
   - `Tensor.amin(dim=None, keepdim=False)`
   - `Tensor.aminmax(dim=None, keepdim=False)`
3. `amin` 支持：
   - 全局规约。
   - 单维规约。
   - 多维 tuple/list 规约。
   - 负维度。
   - `keepdim=True`。
4. `amin` 梯度语义与 PyTorch 对齐：
   - 重复最小值位置平均分配梯度。
   - 多维规约时在所有最小值位置分配梯度。
5. `aminmax` 支持 PyTorch 常用边界：
   - 全局规约。
   - 单个整数 `dim`。
   - `keepdim=True`。
   - 返回具名结果 `min` / `max`。
6. `aminmax` 不实现反向传播，反向时报错，贴近 PyTorch 当前 `aten::aminmax` derivative 未实现的行为。
7. CPU/CUDA 后端一致。

## 实施步骤

1. 改造 `torch_1k/functional/numeric.py`：
   - 新增 `Amin(Function)`。
   - 新增 `AminMax(Function)`。
   - 新增 `amin(...)` 与 `aminmax(...)` 函数。
   - 新增 `AminMaxResult(min, max)` 返回类型。
2. 改造 `torch_1k/tensor.py`：
   - 新增 `Tensor.amin(...)` 方法。
   - 新增 `Tensor.aminmax(...)` 方法。
3. 新增测试：
   - `amin` 全局、tuple dim、负维度、keepdim 前向和梯度与 PyTorch 对比。
   - `Tensor.amin(...)` 方法入口。
   - `aminmax` 全局、单维、keepdim、返回字段与 PyTorch 对比。
   - `aminmax` tuple dim 报错。
   - `aminmax` 反向抛错。
   - CUDA 可选路径。
4. 新增示例：
   - 对 `(N, C, H, W)` 张量一次性提取空间维最小值和全局区间。
   - 同进程对比 `torch_1k` 与 PyTorch 的输出和 `amin` 梯度。
5. 更新 PLAN-033 结果文档和设计路线图。

## 非目标

1. 不实现 `torch.min` / `Tensor.min` 的 values/indices 语义。
2. 不实现 `aminmax` 的 tuple dim，因为 PyTorch 当前只接受单个整数 dim。
3. 不实现 `out=` 参数。
4. 不实现完整 dtype promotion。

## 验收标准

1. 新增 amin/aminmax 测试通过。
2. 既有 amax/max 相关测试不回归。
3. 新增示例与 PyTorch 对比通过。
4. 全量测试通过。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
