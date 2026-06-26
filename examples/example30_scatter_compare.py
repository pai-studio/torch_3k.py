"""Compare scatter and scatter_add behavior with PyTorch in one process."""

import sys
from pathlib import Path

import numpy as np
import torch as pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k


def _compare(op_name, input_data, index_data, src_data, dim):
    ours_input = torch_1k.tensor(input_data, requires_grad=True)
    ours_index = torch_1k.tensor(index_data)
    ours_src = torch_1k.tensor(src_data, requires_grad=True)
    ours_out = getattr(torch_1k, op_name)(
        ours_input, dim, ours_index, ours_src,
    )
    (ours_out * ours_out).sum().backward()

    ref_input = pytorch.tensor(input_data, dtype=pytorch.float64,
                              requires_grad=True)
    ref_index = pytorch.tensor(index_data, dtype=pytorch.int64)
    ref_src = pytorch.tensor(src_data, dtype=pytorch.float64,
                            requires_grad=True)
    ref_out = getattr(pytorch, op_name)(ref_input, dim, ref_index, ref_src)
    (ref_out * ref_out).sum().backward()

    return {
        "out": ours_out.numpy(),
        "out_diff": np.max(np.abs(
            ours_out.numpy() - ref_out.detach().numpy(),
        )),
        "input_grad_diff": np.max(np.abs(
            ours_input.grad.numpy() - ref_input.grad.detach().numpy(),
        )),
        "src_grad_diff": np.max(np.abs(
            ours_src.grad.numpy() - ref_src.grad.detach().numpy(),
        )),
    }


def run():
    one_hot = _compare(
        "scatter",
        np.zeros((3, 4), dtype=np.float64),
        np.array([[0], [2], [1]], dtype=np.int64),
        np.ones((3, 1), dtype=np.float64),
        dim=1,
    )
    bucket_sum = _compare(
        "scatter_add",
        np.zeros((2, 4), dtype=np.float64),
        np.array([[0, 1, 1], [2, 0, 3]], dtype=np.int64),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        dim=1,
    )
    return one_hot, bucket_sum


if __name__ == "__main__":
    one_hot, bucket_sum = run()
    print(f"one_hot=\n{one_hot['out']}")
    print(f"bucket_sum=\n{bucket_sum['out']}")
    print(f"one_hot output diff={one_hot['out_diff']:.12e}")
    print(f"bucket_sum output diff={bucket_sum['out_diff']:.12e}")
    print(f"bucket_sum src grad diff={bucket_sum['src_grad_diff']:.12e}")

    assert np.allclose(one_hot["out"], [[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0]])
    assert np.allclose(bucket_sum["out"], [[1.0, 5.0, 0.0, 0.0],
                                           [5.0, 0.0, 4.0, 6.0]])
    assert one_hot["out_diff"] < 1e-12
    assert one_hot["input_grad_diff"] < 1e-12
    assert one_hot["src_grad_diff"] < 1e-12
    assert bucket_sum["out_diff"] < 1e-12
    assert bucket_sum["input_grad_diff"] < 1e-12
    assert bucket_sum["src_grad_diff"] < 1e-12
