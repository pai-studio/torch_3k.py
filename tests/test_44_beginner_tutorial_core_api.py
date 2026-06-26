import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import torch_1k as torch
import torch_1k.nn as nn
import example38_beginner_tutorial as tutorial


def test_beginner_tensor_wrappers_match_pytorch():
    pytorch = pytest.importorskip("torch")
    data = np.arange(6, dtype=np.float64).reshape(2, 3)

    x = torch.from_numpy(data).to(dtype=torch.float32)
    ref = pytorch.from_numpy(data).to(dtype=pytorch.float32)
    eye = torch.eye(3, dtype=torch.float32)
    ref_eye = pytorch.eye(3, dtype=pytorch.float32)

    actual = x.matmul(eye).sum(dim=1, keepdim=True)
    expected = ref.matmul(ref_eye).sum(dim=1, keepdim=True)
    centered = x - torch.mean(x, dim=0, keepdim=True)
    ref_centered = ref - pytorch.mean(ref, dim=0, keepdim=True)

    assert torch.as_tensor(x) is x
    assert len(x) == len(ref)
    assert x.tolist() == ref.tolist()
    assert np.allclose(actual.numpy(), expected.detach().numpy())
    assert np.allclose(centered.numpy(), ref_centered.detach().numpy())
    assert torch.randn_like(x).shape == x.shape
    assert torch.rand_like(x).dtype == x.dtype


def test_clone_and_mm_are_differentiable_like_pytorch():
    pytorch = pytest.importorskip("torch")
    data = np.arange(6, dtype=np.float64).reshape(2, 3) / 5.0
    weight = np.arange(6, dtype=np.float64).reshape(3, 2) / 7.0

    x = torch.tensor(data, requires_grad=True)
    w = torch.tensor(weight, requires_grad=True)
    ref_x = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    ref_w = pytorch.tensor(weight, dtype=pytorch.float64, requires_grad=True)

    y = x.clone().mm(w).mean()
    ref_y = ref_x.clone().mm(ref_w).mean()
    y.backward()
    ref_y.backward()

    assert np.allclose(y.item(), ref_y.item())
    assert np.allclose(x.grad.numpy(), ref_x.grad.detach().numpy())
    assert np.allclose(w.grad.numpy(), ref_w.grad.detach().numpy())


def test_tanh_sigmoid_modules():
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = nn.Tanh()(x).sum() + nn.Sigmoid()(x).sum()
    y.backward()

    assert y.shape == ()
    assert x.grad.shape == x.shape


def test_torch_1k_beginner_tutorial_runs():
    result = tutorial.run_all()
    assert result["backend"] == "torch_1k"
    assert result["regression_loss"] < 0.03
    assert result["accuracy"] == 1.0


def test_pytorch_beginner_tutorial_replacement_path_runs():
    pytest.importorskip("torch")

    env = os.environ.copy()
    env["USE_TORCH_1K"] = "0"
    script = Path(__file__).resolve().parents[1] / "examples" / (
        "example38_beginner_tutorial.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        "PyTorch replacement path failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
