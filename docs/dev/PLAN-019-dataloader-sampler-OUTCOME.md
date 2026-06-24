# PLAN-019 DataLoader sampler / batch_sampler 结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-019-dataloader-sampler.md`

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 实施结果

1. 新增 `torch_1k.utils.data.sampler`：
   - `Sampler`
   - `SequentialSampler`
   - `RandomSampler`
   - `SubsetRandomSampler`
   - `BatchSampler`
2. `DataLoader` 新增参数：
   - `sampler=None`
   - `batch_sampler=None`
3. `DataLoader` 默认行为保持兼容：
   - `shuffle=False` 默认使用 `SequentialSampler`。
   - `shuffle=True` 默认使用 `RandomSampler`。
   - `batch_size` / `drop_last` 由 `BatchSampler` 统一处理。
4. `DataLoader` 参数互斥校验：
   - `sampler` 与 `shuffle=True` 互斥。
   - `batch_sampler` 与 `batch_size`、`shuffle`、`sampler`、`drop_last` 互斥。
5. `DataLoader.__len__()` 现在返回 `batch_sampler` 长度。
6. `torch_1k.utils.data.__init__` 已导出新增 sampler 类。
7. 新增 `examples/example19_dataloader_sampler_compare.py`：
   - 覆盖 `SequentialSampler`。
   - 覆盖 `BatchSampler`。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
8. 更新设计文档，记录 DataLoader sampler / batch_sampler 已补齐。

## 新增测试

新增 `tests/test_28_dataloader_sampler.py`，覆盖：

1. `SequentialSampler` 顺序 batch。
2. `RandomSampler` 受 `torch.manual_seed()` 控制，并覆盖全部样本。
3. `SubsetRandomSampler` 只采样指定子集。
4. `BatchSampler(drop_last=True)`。
5. `DataLoader(batch_sampler=...)` 自定义 batch 顺序。
6. sampler / batch_sampler 参数互斥校验。
7. `torch.utils.data` 命名空间导出 sampler 类。

## 验证结果

已运行：

```bash
pytest -q tests/test_28_dataloader_sampler.py
python examples/example19_dataloader_sampler_compare.py
USE_TORCH_1K=0 python examples/example19_dataloader_sampler_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 DataLoader sampler 测试：`7 passed`
- DataLoader sampler 双后端示例：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`125 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现多进程 `num_workers`。
2. 尚未实现 pinned memory。
3. 尚未实现 distributed sampler。

## 复核结论

本轮补齐了真实训练数据管线中常见的采样控制能力。`DataLoader` 现在可以表达顺序采样、随机采样、子集随机采样和自定义 batch 采样，同时保持旧的 `batch_size` / `shuffle` 使用方式不变。
