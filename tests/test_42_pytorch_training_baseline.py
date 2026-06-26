import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import example36_pytorch_training_baseline as baseline


def _assert_baseline_results(results):
    assert results["mlp"]["accuracy"] >= 0.95
    assert results["cnn"]["accuracy"] >= 0.95
    assert results["transformer"]["accuracy"] >= 0.875
    assert results["mlp"]["checkpoint_error"] <= 1e-5


def test_torch_1k_training_baseline_runs():
    results = baseline.run_all()
    assert results["backend"] == "torch_1k"
    _assert_baseline_results(results)


def test_pytorch_import_replacement_path_runs():
    pytest.importorskip("torch")

    env = os.environ.copy()
    env["USE_TORCH_1K"] = "0"
    script = Path(__file__).resolve().parents[1] / "examples" / (
        "example36_pytorch_training_baseline.py"
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
