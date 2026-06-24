"""PyTorch-compatible mini-batch CNN training with TensorDataset/DataLoader."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    import torch_1k.nn as nn
    import torch_1k.optim as optim
    from torch_1k.utils.data import DataLoader, TensorDataset
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

from example5_mnist_cnn_train_compare import SmallCNN, make_dataset


def run(epochs=100, lr=0.02, batch_size=5):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = make_dataset(device)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == y).float().mean().item()
    return loss.item(), accuracy, pred.detach().cpu().numpy()


if __name__ == "__main__":
    final_loss, accuracy, pred = run()
    print(f"final_loss={final_loss:.6f}")
    print(f"accuracy={accuracy:.6f}")
    print(f"pred={pred}")
    assert accuracy >= 0.95
