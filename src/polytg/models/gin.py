"""Configurable edge-aware Graph Isomorphism Network regressor."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class GINRegressor(nn.Module):
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
        self.convolutions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            update = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convolutions.append(GINEConv(update, edge_dim=edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
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
        for convolution, batch_norm in zip(
            self.convolutions, self.batch_norms, strict=True
        ):
            residual = x
            x = convolution(x, edge_index, edge_attr)
            if x.shape[0] > 1 or not self.training:
                x = batch_norm(x)
            x = F.relu(x + residual)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, batch)).squeeze(-1)
