# PLAN-034 amin / aminmax 规约 API 结果

日期：2026-06-26

计划文件：`docs/dev/PLAN-034-amin-aminmax-api.md`

Git 基线：`c125f9d`

## 实施结果

1. 新增顶层函数：
   - `torch_1k.amin(input, dim=None, keepdim=False)`
   - `torch_1k.aminmax(input, dim=None, keepdim=False)`
2. 新增 Tensor 方法：
   - `Tensor.amin(dim=None, keepdim=False)`
   - `Tensor.aminmax(dim=None, keepdim=False)`
3. 新增返回类型：
   - `AminMaxResult(min, max)`
4. `amin` 支持：
   - 全局规约。
   - 单维规约。
   - 多维 tuple/list 规约。
   - 负维度。
   - `keepdim=True`。
   - `axis` / `keepdims` 兼容别名。
5. `amin` 梯度与 PyTorch 对齐：
   - 重复最小值位置平均分配梯度。
   - 多维 tuple 规约时在所有最小值位置分配梯度。
6. `aminmax` 支持：
   - 全局规约。
   - 单个整数 `dim`。
   - `keepdim=True`。
   - `axis` / `keepdims` 兼容别名。
   - 返回 `.min` / `.max` 字段。
7. `aminmax` 反向传播显式抛出 `RuntimeError`，贴近 PyTorch 当前 `aten::aminmax` derivative 未实现的边界。
8. CPU/CUDA 共用同一套实现。

## 新增测试

新增 `tests/test_40_amin_aminmax_api.py`，覆盖：

1. 全局 `amin` 的重复最小值梯度平均分配。
2. `amin(dim=(...))` tuple 维度规约前向和梯度与 PyTorch 对比。
3. `keepdim=True`。
4. 负维度。
5. `Tensor.amin(...)` 方法入口。
6. 非法重复维度、越界维度和非法 dim 类型报错。
7. `aminmax` 全局规约返回 `.min` / `.max`。
8. `aminmax(dim=..., keepdim=True)` 与 PyTorch 对比。
9. `aminmax` 拒绝 tuple dim。
10. `aminmax` 反向抛错。
11. CUDA 可选路径下输出和梯度保持 CUDA 设备。

## 新增示例

新增 `examples/example34_amin_aminmax_compare.py`：

1. 用 `(N, C, H, W)` 张量做空间维 `(H, W)` 最小值规约。
2. 对比 `torch_1k.amin` 与 PyTorch 的输出和梯度。
3. 使用 `torch_1k.aminmax(..., dim=1, keepdim=True)` 提取通道维区间。
4. 对比 `aminmax` 的 `.min` / `.max` 输出。

## 验证结果

已运行：

```bash
pytest -q tests/test_40_amin_aminmax_api.py
pytest -q tests/test_21_max_api.py tests/test_39_amax_api.py tests/test_40_amin_aminmax_api.py
python examples/example34_amin_aminmax_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 amin/aminmax 测试：`10 passed`
- max/amax/amin/aminmax 相关测试：`22 passed`
- 新增示例：通过，输出和 `amin` 梯度均与 PyTorch 一致
- 全量测试：`234 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. `torch.min` / `Tensor.min` 的 values/indices 语义已在 PLAN-035 补齐。
2. 尚未实现 `out=` 参数。
3. 尚未实现完整 dtype promotion。

## 复核结论

本轮补齐了 values-only 最小值规约和一次性区间提取 API。现在用户可以用 `torch_1k.amin(x, dim=(...))` 表达多维最小值聚合，也可以用 `torch_1k.aminmax(x, dim=...)` 获取 PyTorch 风格的 `.min` / `.max` 结果；反向传播边界对齐 PyTorch 当前行为，没有把 `aminmax` 包装成不同语义的可微接口。
