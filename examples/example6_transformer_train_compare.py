"""PyTorch-compatible Transformer encoder example.

The task is a tiny sequence classification problem where the first token
determines the class. It verifies Embedding, TransformerEncoderLayer, LayerNorm,
attention, CrossEntropyLoss, and Adam all work together.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    import torch_1k.nn as nn
    import torch_1k.optim as optim
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim


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


def make_dataset(device):
    x = np.array([
        [0, 2, 3, 4, 5],
        [1, 3, 4, 5, 6],
        [0, 4, 5, 6, 7],
        [1, 5, 6, 7, 8],
        [0, 6, 7, 8, 9],
        [1, 7, 8, 9, 10],
        [0, 8, 9, 10, 11],
        [1, 2, 5, 8, 11],
    ])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return torch.tensor(x).long().to(device), torch.tensor(y).long().to(device)


def run(epochs=160, lr=0.03):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = make_dataset(device)

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
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == y).float().mean().item()
    return loss.item(), accuracy, pred.detach().cpu().numpy()


if __name__ == "__main__":
    final_loss, accuracy, pred = run()
    print(f"final_loss={final_loss:.6f}")
    print(f"accuracy={accuracy:.6f}")
    print(f"pred={pred}")
    assert accuracy == 1.0
