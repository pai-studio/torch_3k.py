# PLAN-019 DataLoader sampler / batch_sampler

日期：2026-06-24

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 目标

补齐 PyTorch 风格数据加载中常用的 sampler / batch_sampler 能力，使 `DataLoader` 可以表达固定顺序、随机顺序、子集采样和自定义 batch 采样。

本计划聚焦单进程 map-style dataset，不实现多 worker、pinned memory 或完整 PyTorch DataLoader 参数体系。

## 当前缺口

1. 设计文档记录 `DataLoader` 仍缺少 sampler / 更完整参数兼容。
2. 当前 `DataLoader` 只支持 `shuffle=True/False`，不能表达固定子集、指定采样顺序或自定义 batch。
3. 真实训练和调试中常用：

```python
DataLoader(dataset, sampler=SubsetRandomSampler(indices), batch_size=...)
DataLoader(dataset, batch_sampler=BatchSampler(sampler, batch_size, drop_last))
```

4. `torch_1k.utils.data` 当前没有 `SequentialSampler` / `RandomSampler` / `BatchSampler` 等基础类。

## 实施步骤

1. 新增 `torch_1k/utils/data/sampler.py`：
   - `Sampler`
   - `SequentialSampler`
   - `RandomSampler`
   - `SubsetRandomSampler`
   - `BatchSampler`
2. 改造 `DataLoader`：
   - 支持 `sampler=None`。
   - 支持 `batch_sampler=None`。
   - 默认行为保持与当前一致：`shuffle=True` 使用随机采样，否则顺序采样。
   - `sampler` 与 `shuffle=True` 互斥。
   - `batch_sampler` 与 `batch_size/shuffle/sampler/drop_last` 互斥。
   - `__len__()` 返回 batch_sampler 长度。
3. 更新 `torch_1k.utils.data.__init__` 导出 sampler 类。
4. 新增 `examples/example19_dataloader_sampler_compare.py`，覆盖 PyTorch import 切换路径。
5. 新增 `tests/test_28_dataloader_sampler.py`，覆盖：
   - SequentialSampler 顺序 batch。
   - RandomSampler 覆盖全部样本且受 `manual_seed` 控制。
   - SubsetRandomSampler 子集采样。
   - BatchSampler drop_last。
   - DataLoader 使用自定义 batch_sampler。
   - 参数互斥校验。
6. 更新设计文档记录 PLAN-019。
7. 运行新增测试、示例、全量测试和编译检查。

## 验收标准

1. `DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=2)` 可用。
2. `DataLoader(dataset, sampler=SubsetRandomSampler(indices), batch_size=2)` 可用。
3. `DataLoader(dataset, batch_sampler=BatchSampler(...))` 可用。
4. `shuffle=True` 默认随机采样行为保持可用。
5. 旧 DataLoader 测试继续通过。
6. 全量测试通过。

## TODO 记录

1. 本计划不实现多进程 `num_workers`。
2. 本计划不实现 pinned memory。
3. 本计划不实现 distributed sampler。
