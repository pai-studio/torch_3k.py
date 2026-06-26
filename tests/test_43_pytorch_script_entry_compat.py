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
import torch_1k.optim as optim
import example37_pytorch_script_entry_compat as entry_compat


def test_device_to_dtype_and_tensor_metadata():
    device = torch.device("cpu")
    x = torch.randn(2, 3, device=device, dtype=torch.float32,
                    requires_grad=True)
    y = x.to(dtype=torch.float64)
    z = y.to(x)

    assert str(device) == "cpu"
    assert x.dtype == np.float32
    assert x.dim() == 2
    assert x.numel() == 6
    assert x.is_cuda is False
    assert y.dtype == np.float64
    assert z.dtype == x.dtype
    assert z.device == x.device
    assert z.requires_grad is True


def test_module_iteration_modulelist_identity_and_zero_grad():
    class SmallModuleListModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = nn.Identity()
            self.layers = nn.ModuleList([
                nn.Linear(2, 4),
                nn.ReLU(),
                nn.Linear(4, 2),
            ])

        def forward(self, x):
            x = self.identity(x)
            for layer in self.layers:
                x = layer(x)
            return x

    model = SmallModuleListModel()

    names = [name for name, _ in model.named_modules()]
    children = list(model.children())
    module_list = model.layers

    assert len(children) == 2
    assert len(module_list) == 3
    assert isinstance(module_list[0], nn.Linear)
    assert names == [
        "",
        "identity",
        "layers",
        "layers.0",
        "layers.1",
        "layers.2",
    ]

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    optimizer.zero_grad(set_to_none=False)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(np.allclose(parameter.grad.numpy(), 0.0)
               for parameter in model.parameters())

    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_torch_1k_script_entry_example_runs():
    result = entry_compat.run()
    assert result["backend"] == "torch_1k"
    assert result["accuracy"] == 1.0
    assert "layers.0" in result["named_modules"]


def test_pytorch_script_entry_replacement_path_runs():
    pytest.importorskip("torch")

    env = os.environ.copy()
    env["USE_TORCH_1K"] = "0"
    script = Path(__file__).resolve().parents[1] / "examples" / (
        "example37_pytorch_script_entry_compat.py"
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
