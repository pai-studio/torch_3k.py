"""einsum 经典用法对照示例。

本示例同时运行 torch_1k 和 PyTorch，逐项比较前向结果与梯度。
覆盖矩阵乘、batch matmul、点积、外积、规约、attention 和广播门控。
"""

import sys
from pathlib import Path

import numpy as np
import torch as torch_ref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k as torch_1k


def _array(shape, offset=0.0):
    size = int(np.prod(shape))
    return (np.arange(size, dtype=np.float64).reshape(shape) + offset) / 7.0


def _to_numpy(x):
    return x.detach().cpu().numpy()


def _max_abs_diff(a, b):
    if a.shape == () and b.shape == ():
        return float(abs(a.item() - b.item()))
    return float(np.max(np.abs(a - b)))


def compare_case(name, equation, arrays):
    ours = [torch_1k.tensor(array, requires_grad=True) for array in arrays]
    refs = [
        torch_ref.tensor(array, dtype=torch_ref.float64, requires_grad=True)
        for array in arrays
    ]

    ours_out = torch_1k.einsum(equation, *ours)
    refs_out = torch_ref.einsum(equation, *refs)
    ours_out.sum().backward()
    refs_out.sum().backward()

    value_diff = _max_abs_diff(_to_numpy(ours_out), _to_numpy(refs_out))
    grad_diffs = [
        _max_abs_diff(_to_numpy(ours_x.grad), _to_numpy(ref_x.grad))
        for ours_x, ref_x in zip(ours, refs)
    ]
    return {
        "name": name,
        "equation": equation,
        "shape": tuple(ours_out.shape),
        "value_diff": value_diff,
        "grad_diffs": grad_diffs,
    }


def compare_attention_pipeline():
    query = _array((2, 3, 4), 1.0)
    key = _array((2, 5, 4), 2.0)
    value = _array((2, 5, 6), 3.0)

    ours_q = torch_1k.tensor(query, requires_grad=True)
    ours_k = torch_1k.tensor(key, requires_grad=True)
    ours_v = torch_1k.tensor(value, requires_grad=True)
    refs_q = torch_ref.tensor(query, dtype=torch_ref.float64, requires_grad=True)
    refs_k = torch_ref.tensor(key, dtype=torch_ref.float64, requires_grad=True)
    refs_v = torch_ref.tensor(value, dtype=torch_ref.float64, requires_grad=True)

    ours_scores = torch_1k.einsum("bqd,bkd->bqk", ours_q, ours_k)
    ours_context = torch_1k.einsum("bqk,bkd->bqd", ours_scores, ours_v)
    refs_scores = torch_ref.einsum("bqd,bkd->bqk", refs_q, refs_k)
    refs_context = torch_ref.einsum("bqk,bkd->bqd", refs_scores, refs_v)

    ours_context.sum().backward()
    refs_context.sum().backward()

    return {
        "name": "attention score/context",
        "equation": "bqd,bkd->bqk; bqk,bkd->bqd",
        "shape": tuple(ours_context.shape),
        "value_diff": _max_abs_diff(_to_numpy(ours_context),
                                    _to_numpy(refs_context)),
        "grad_diffs": [
            _max_abs_diff(_to_numpy(ours_q.grad), _to_numpy(refs_q.grad)),
            _max_abs_diff(_to_numpy(ours_k.grad), _to_numpy(refs_k.grad)),
            _max_abs_diff(_to_numpy(ours_v.grad), _to_numpy(refs_v.grad)),
        ],
    }


def run():
    cases = [
        ("矩阵乘", "ij,jk->ik", [_array((2, 3), 1.0), _array((3, 4), 2.0)]),
        ("批量矩阵乘", "bij,bjk->bik",
         [_array((2, 3, 4), 1.0), _array((2, 4, 5), 2.0)]),
        ("点积", "i,i->", [_array((4,), 1.0), _array((4,), 2.0)]),
        ("外积", "i,j->ij", [_array((3,), 1.0), _array((4,), 2.0)]),
        ("行规约", "ij->i", [_array((2, 3), 1.0)]),
        ("全局规约", "ij->", [_array((2, 3), 1.0)]),
        ("逐元素乘加", "ij,ij->",
         [_array((2, 3), 1.0), _array((2, 3), 2.0)]),
        ("广播门控", "ij,j->ij",
         [_array((2, 1), 1.0), _array((3,), 2.0)]),
        ("广播乘后规约", "ij,j->i",
         [_array((2, 1), 1.0), _array((3,), 2.0)]),
    ]
    results = [
        compare_case(name, equation, arrays)
        for name, equation, arrays in cases
    ]
    results.append(compare_attention_pipeline())
    return results


if __name__ == "__main__":
    results = run()
    for item in results:
        max_grad_diff = max(item["grad_diffs"]) if item["grad_diffs"] else 0.0
        print(
            f"{item['name']}: {item['equation']}, shape={item['shape']}, "
            f"value_diff={item['value_diff']:.3e}, "
            f"max_grad_diff={max_grad_diff:.3e}"
        )
        assert item["value_diff"] < 1e-10
        assert max_grad_diff < 1e-10
