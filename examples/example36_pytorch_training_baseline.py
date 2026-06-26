"""PyTorch-compatible training baseline.

This example keeps the training bodies backend-neutral. Set USE_TORCH_1K=0 to
run the same workloads with PyTorch instead of torch_1k.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    import torch_1k.nn as nn
    import torch_1k.optim as optim
    from torch_1k.utils.data import DataLoader, TensorDataset

    BACKEND_NAME = "torch_1k"
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    BACKEND_NAME = "torch"


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(12, 3),
        )

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 3),
        )

    def forward(self, x):
        return self.net(x)


class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size=12, d_model=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=16,
            batch_first=True,
        )
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])


def _seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean().item(), _to_numpy(pred)


def make_tabular_dataset(device):
    rng = np.random.RandomState(0)
    centers = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [0.0, 1.1],
    ], dtype="float32")
    features = []
    labels = []
    for label, center in enumerate(centers):
        for _ in range(12):
            features.append(center + rng.normal(0, 0.05, size=2))
            labels.append(label)
    x = torch.tensor(np.array(features, dtype="float32")).to(device)
    y = torch.tensor(np.array(labels, dtype="int64")).long().to(device)
    return TensorDataset(x, y)


def make_image_dataset(device):
    rng = np.random.RandomState(1)
    anchors = [(1, 1), (1, 5), (5, 2)]
    images = []
    labels = []
    for label, (row, col) in enumerate(anchors):
        for _ in range(8):
            image = rng.normal(0, 0.02, size=(8, 8)).astype("float32")
            image[row:row + 2, col:col + 2] += 1.0
            image[row + 1, :] += 0.12
            image[:, col + 1] += 0.12
            images.append(image)
            labels.append(label)
    x = torch.tensor(np.array(images, dtype="float32")[:, None, :, :]).to(device)
    y = torch.tensor(np.array(labels, dtype="int64")).long().to(device)
    return x, y


def make_sequence_dataset(device):
    x = np.array([
        [0, 2, 3, 4, 5],
        [1, 3, 4, 5, 6],
        [0, 4, 5, 6, 7],
        [1, 5, 6, 7, 8],
        [0, 6, 7, 8, 9],
        [1, 7, 8, 9, 10],
        [0, 8, 9, 10, 11],
        [1, 2, 5, 8, 11],
    ], dtype="int64")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="int64")
    return torch.tensor(x).long().to(device), torch.tensor(y).long().to(device)


def _checkpoint_roundtrip(model_factory, model, sample, device):
    model.eval()
    with torch.no_grad():
        expected = _to_numpy(model(sample))

    path = Path(tempfile.gettempdir()) / (
        f"torch_3k_plan36_{BACKEND_NAME}_{os.getpid()}.pkl"
    )
    torch.save(model.state_dict(), path)
    restored = model_factory().to(device)
    restored.load_state_dict(torch.load(path))
    restored.eval()
    try:
        path.unlink()
    except OSError:
        pass

    with torch.no_grad():
        actual = _to_numpy(restored(sample))
    return float(np.max(np.abs(expected - actual)))


def run_mlp(device, epochs=70, lr=0.03):
    dataset = make_tabular_dataset(device)
    loader = DataLoader(dataset, batch_size=9, shuffle=True)
    x, y = dataset.tensors

    model = TinyMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
        accuracy, pred = _accuracy(logits, y)

    checkpoint_error = _checkpoint_roundtrip(TinyMLP, model, x, device)
    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "pred": pred,
        "checkpoint_error": checkpoint_error,
    }


def run_cnn(device, epochs=70, lr=0.02):
    x, y = make_image_dataset(device)
    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
        accuracy, pred = _accuracy(logits, y)

    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "pred": pred,
    }


def run_transformer(device, epochs=120, lr=0.03):
    x, y = make_sequence_dataset(device)
    model = TinyTransformerClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    for _ in range(epochs):
        model.train()
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
        accuracy, pred = _accuracy(logits, y)

    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "pred": pred,
    }


def run_all():
    _seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "backend": BACKEND_NAME,
        "device": device,
        "mlp": run_mlp(device),
        "cnn": run_cnn(device),
        "transformer": run_transformer(device),
    }


def _print_results(results):
    print(f"backend={results['backend']}")
    print(f"device={results['device']}")
    for name in ("mlp", "cnn", "transformer"):
        result = results[name]
        print(
            f"{name}: loss={result['loss']:.6f}, "
            f"accuracy={result['accuracy']:.6f}, pred={result['pred']}"
        )
    print(
        "mlp_checkpoint_error="
        f"{results['mlp']['checkpoint_error']:.8f}"
    )


if __name__ == "__main__":
    results = run_all()
    _print_results(results)
    assert results["mlp"]["accuracy"] >= 0.95
    assert results["cnn"]["accuracy"] >= 0.95
    assert results["transformer"]["accuracy"] >= 0.875
    assert results["mlp"]["checkpoint_error"] <= 1e-5
