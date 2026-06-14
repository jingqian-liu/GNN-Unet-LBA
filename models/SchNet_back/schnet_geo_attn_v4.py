#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
SchNet with low-rank / group-wise geometric attention on messages (v4).

Based on schnet_geo_attn.py. No residual connection, no edge-type embedding.

Full channel-wise gating is expressive but has many parameters and can overfit.
Group-wise gating is a middle ground: the gate outputs G values and each value
controls a group of (num_filters // G) contiguous channels:

    gate_group = σ( gate_mlp([h_i, h_j, RBF(d_ij)]) )        # [E, G]
    gate       = gate_group.repeat_interleave(F // G, dim=-1)  # [E, F]
    m_ij       = gate ⊙ W(d_ij) ⊙ lin1(h_j)                   # [E, F]

With hidden=64 and G=8: 8 groups of 8 channels; gate_mlp is much smaller
than full channel-wise while still allowing per-group modulation.

num_filters is set equal to hidden_size in the encoder so group sizes are
natural (e.g. 64 // 8 = 8 channels per group).
"""

from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing

from .schnet import ShiftedSoftplus, SchNet


class GroupAttnCFConv(MessagePassing):
    """CFConv with group-wise geometric attention (low-rank gate).

    Gate uses raw (pre-linear) node features:
        gate_group = σ( gate_mlp([h_i, h_j, rbf]) )           # [E, n_groups]
        gate       = repeat_interleave(gate_group, F//G)       # [E, num_filters]
        m_ij       = gate ⊙ W(d_ij) ⊙ lin1(h_j)               # [E, num_filters]

    Requires: num_filters % n_groups == 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_gaussians: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
        n_groups: int = 8,
    ):
        super().__init__(aggr='add')
        assert num_filters % n_groups == 0, (
            f'num_filters ({num_filters}) must be divisible by n_groups ({n_groups})'
        )
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn          # filter MLP: RBF -> num_filters (unchanged)
        self.cutoff = cutoff
        self.n_groups = n_groups
        self.group_size = num_filters // n_groups

        self.gate_mlp = Sequential(
            Linear(2 * in_channels + num_gaussians, n_groups * 4),
            ShiftedSoftplus(),
            Linear(n_groups * 4, n_groups),   # [E, n_groups]
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
        gate_group = torch.sigmoid(
            self.gate_mlp(torch.cat([x_i, x_j, rbf], dim=-1))
        )                                                          # [E, G]
        gate = gate_group.repeat_interleave(self.group_size, dim=-1)  # [E, F]
        h_j_proj = self.lin1(x_j)                                 # [E, F]
        return gate * W * h_j_proj                                 # [E, F]


class GroupAttnInteractionBlock(torch.nn.Module):
    """Drop-in replacement for InteractionBlock using GroupAttnCFConv."""

    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float, n_groups: int = 8):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = GroupAttnCFConv(
            hidden_channels, hidden_channels, num_gaussians,
            num_filters, self.mlp, cutoff, n_groups=n_groups,
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


class SchNetGeoAttnV4(SchNet):
    """SchNet with group-wise geometric attention messages.

    Identical to SchNet except interactions use GroupAttnInteractionBlock.
    Inherits forward(), distance_expansion, and all other methods unchanged.
    """

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, n_groups: int = 8, **kwargs):
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            **kwargs,
        )
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(
                GroupAttnInteractionBlock(
                    hidden_channels, num_gaussians, num_filters, cutoff,
                    n_groups=n_groups,
                )
            )


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

from torch_scatter import scatter_mean, scatter_sum


class SchNetGeoAttnV4Encoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3,
                 n_groups: int = 8) -> None:
        super().__init__()

        self.num_gaussians = 50

        # Set num_filters = hidden_size so group sizes are natural:
        # e.g. hidden=64, G=8 → 8 groups of 8 channels each
        self.encoder = SchNetGeoAttnV4(
            hidden_size,
            num_filters=hidden_size,
            num_interactions=n_layers,
            num_gaussians=self.num_gaussians,
            n_groups=n_groups,
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
