#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
SchNet with geometric attention on messages.

Only CFConv and InteractionBlock are changed vs. the original schnet.py.
All other components (GaussianSmearing, ShiftedSoftplus, SchNet backbone
structure, RadiusInteractionGraph) are imported from schnet.py unchanged.

Original message:
    m_ij = h_j ⊙ W(d_ij)           W = filter_mlp(RBF(d_ij))

New message:
    a_ij = σ( gate_mlp([h_i, h_j, RBF(d_ij)]) )
    m_ij = a_ij ⊙ W(d_ij) ⊙ h_j
"""

from math import pi as PI

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing

from .schnet import GaussianSmearing, ShiftedSoftplus, SchNet


class GeoAttnCFConv(MessagePassing):
    """CFConv with gated attention on messages.

    Gate uses raw (pre-linear) node features so it matches the pseudocode:
        h_i = h[row], h_j = h[col]  (original features)
        a_ij = σ( gate_mlp([h_i, h_j, rbf]) )
        m_ij = a_ij * filter_mlp(rbf) * lin1(h_j)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_gaussians: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn          # filter MLP: RBF -> num_filters (same as original)
        self.cutoff = cutoff

        self.gate_mlp = Sequential(
            Linear(2 * in_channels + num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        self.gate_mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.gate_mlp[2].weight)
        self.gate_mlp[2].bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)           # [E, num_filters]

        # Pass raw x and rbf to message (do NOT pre-apply lin1)
        out = self.propagate(edge_index, x=x, W=W, rbf=edge_attr)
        out = self.lin2(out)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, W: Tensor,
                rbf: Tensor) -> Tensor:
        a_ij = torch.sigmoid(
            self.gate_mlp(torch.cat([x_i, x_j, rbf], dim=-1))
        )                                                 # [E, num_filters]
        return a_ij * W * self.lin1(x_j)                 # [E, num_filters]


class GeoAttnInteractionBlock(torch.nn.Module):
    """Drop-in replacement for InteractionBlock using GeoAttnCFConv."""

    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = GeoAttnCFConv(
            hidden_channels, hidden_channels, num_gaussians,
            num_filters, self.mlp, cutoff,
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetGeoAttn(SchNet):
    """SchNet backbone with geometric attention messages.

    Identical to SchNet except interactions use GeoAttnInteractionBlock.
    Inherits forward(), distance_expansion, and all other methods unchanged.
    """

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, **kwargs):
        # Call SchNet.__init__ to set up everything (lin1/lin2, smearing, etc.)
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            **kwargs,
        )
        # Replace the interaction blocks built by SchNet.__init__
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(
                GeoAttnInteractionBlock(
                    hidden_channels, num_gaussians, num_filters, cutoff,
                )
            )


# ---------------------------------------------------------------------------
# Encoder wrapper (mirrors models/SchNet/encoder.py for SchNetGeoAttn)
# ---------------------------------------------------------------------------

import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum


class SchNetGeoAttnEncoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3) -> None:
        super().__init__()

        self.num_gaussians = 50

        self.encoder = SchNetGeoAttn(
            hidden_size, num_interactions=n_layers,
            num_gaussians=self.num_gaussians,
        )

        self.edge_linear = (
            nn.Linear(edge_size, self.num_gaussians)
            if edge_size != 0 else None
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H = scatter_mean(H, block_id, dim=0)
        Z = scatter_mean(Z, block_id, dim=0).squeeze()
        if edge_attr is not None and self.edge_linear is not None:
            edge_attr = self.edge_linear(edge_attr)
        block_repr = self.encoder(H, Z, batch_id, edges, edge_attr)
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None
