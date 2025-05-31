from __future__ import annotations
"""Placeholder GNN utilities using PyTorch Geometric (optional).

The code does **not** import torch-geometric unless the caller explicitly
invokes `train_gnn` – this keeps startup light and optional.
"""

from typing import Optional

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Data = None  # type: ignore
    GCNConv = None  # type: ignore


def train_gnn(edge_index, num_nodes: int, epochs: int = 10):  # noqa: D401
    """Very small 2-layer GCN for demo purposes."""
    if torch is None:
        print("PyTorch Geometric not installed – skipping GNN training.")
        return None

    x = torch.eye(num_nodes)
    data = Data(x=x, edge_index=edge_index)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_nodes, 16)
            self.conv2 = GCNConv(16, 2)

        def forward(self, d):
            h = self.conv1(d.x, d.edge_index).relu()
            h = self.conv2(h, d.edge_index)
            return h

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = out.mean()
        loss.backward()
        optimizer.step()
    return model 