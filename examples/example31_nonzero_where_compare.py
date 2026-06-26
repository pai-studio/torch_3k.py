"""Compare nonzero and where(condition) behavior with PyTorch."""

import sys
from pathlib import Path

import numpy as np
import torch as pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k


def run():
    scores_data = np.array([
        [0.0, 2.0, -1.0, 0.0],
        [3.0, 0.0, 0.0, 4.0],
    ])
    mask_data = scores_data > 0

    ours_scores = torch_1k.tensor(scores_data)
    ours_mask = ours_scores > 0
    ours_coords = torch_1k.nonzero(ours_mask)
    ours_rows, ours_cols = torch_1k.where(ours_mask)
    ours_numeric_coords = ours_scores.nonzero()

    ref_scores = pytorch.tensor(scores_data, dtype=pytorch.float64)
    ref_mask = ref_scores > 0
    ref_coords = pytorch.nonzero(ref_mask)
    ref_rows, ref_cols = pytorch.where(ref_mask)
    ref_numeric_coords = ref_scores.nonzero()

    return {
        "mask": mask_data,
        "coords": ours_coords.numpy(),
        "coords_diff": np.max(np.abs(
            ours_coords.numpy() - ref_coords.numpy(),
        )),
        "rows_diff": np.max(np.abs(ours_rows.numpy() - ref_rows.numpy())),
        "cols_diff": np.max(np.abs(ours_cols.numpy() - ref_cols.numpy())),
        "numeric_diff": np.max(np.abs(
            ours_numeric_coords.numpy() - ref_numeric_coords.numpy(),
        )),
    }


if __name__ == "__main__":
    result = run()
    print(f"mask=\n{result['mask']}")
    print(f"coords=\n{result['coords']}")
    print(f"coords_diff={result['coords_diff']:.12e}")
    print(f"rows_diff={result['rows_diff']:.12e}")
    print(f"cols_diff={result['cols_diff']:.12e}")
    print(f"numeric_diff={result['numeric_diff']:.12e}")

    assert np.allclose(result["coords"], [[0, 1], [1, 0], [1, 3]])
    assert result["coords_diff"] == 0.0
    assert result["rows_diff"] == 0.0
    assert result["cols_diff"] == 0.0
    assert result["numeric_diff"] == 0.0
