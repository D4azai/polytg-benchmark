"""Configurable Graph Convolutional Network regressor."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNRegressor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.dropout = dropout
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_weight = nn.Sequential(nn.Linear(edge_dim, 1), nn.Softplus())
        self.convolutions = nn.ModuleList(
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_layers))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_encoder(x)
        weights = self.edge_weight(edge_attr).squeeze(-1) if edge_attr.numel() else None
        for convolution, batch_norm in zip(
            self.convolutions, self.batch_norms, strict=True
        ):
            x = convolution(x, edge_index, edge_weight=weights)
            if x.shape[0] > 1 or not self.training:
                x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, batch)).squeeze(-1)
