#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
SchNet with residual geometry-aware gate on messages (v2).

Only CFConv and InteractionBlock are changed vs. the original schnet.py.
All shared utilities are imported from schnet.py unchanged.

Original message:
    m_ij = h_j ⊙ W(d_ij)

New message:
    g_ij  = tanh( gate_mlp([h_i, h_j, RBF(d_ij)]) )   # scalar [E, 1]
    m_ij  = (1 + ε · g_ij) · W(d_ij) · lin1(h_j)

ε is a learnable scalar initialised to 0, so at the start of training
the model is exactly equivalent to vanilla SchNet.
"""

from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing

from .schnet import GaussianSmearing, ShiftedSoftplus, SchNet


class ResGeoAttnCFConv(MessagePassing):
    """CFConv with residual geometry-aware scalar gate.

    Gate uses raw (pre-linear) node features:
        h_i = h[row],  h_j = h[col]   (original features, not post-lin1)
        g_ij  = tanh( gate_mlp([h_i, h_j, rbf]) )        # [E, 1]
        gate  = 1 + eps * g_ij                            # residual
        m_ij  = gate * W(d_ij) * lin1(h_j)
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
        self.nn = nn          # filter MLP: RBF -> num_filters (unchanged)
        self.cutoff = cutoff

        self.gate_mlp = Sequential(
            Linear(2 * in_channels + num_gaussians, in_channels),
            ShiftedSoftplus(),
            Linear(in_channels, 1),   # scalar gate — more stable
        )

        # Residual scale: init 0 so model starts identical to SchNet
        # (use torch.nn explicitly — local param `nn` shadows the module import)
        self.eps = torch.nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        self.gate_mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.gate_mlp[2].weight)
        self.gate_mlp[2].bias.data.fill_(0)
        # eps stays at 0 after reset

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
        g = torch.tanh(
            self.gate_mlp(torch.cat([x_i, x_j, rbf], dim=-1))
        )                                                 # [E, 1]
        gate = 1.0 + self.eps * g                        # [E, 1], residual
        h_j_proj = self.lin1(x_j)                        # [E, num_filters]
        return gate * W * h_j_proj                        # [E, num_filters]


class ResGeoAttnInteractionBlock(torch.nn.Module):
    """Drop-in replacement for InteractionBlock using ResGeoAttnCFConv."""

    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = ResGeoAttnCFConv(
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


class SchNetGeoAttnV2(SchNet):
    """SchNet with residual geometry-aware gate messages.

    Identical to SchNet except interactions use ResGeoAttnInteractionBlock.
    Inherits forward(), distance_expansion, and all other methods unchanged.
    """

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, **kwargs):
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
                ResGeoAttnInteractionBlock(
                    hidden_channels, num_gaussians, num_filters, cutoff,
                )
            )


# ---------------------------------------------------------------------------
# Encoder wrapper (mirrors SchNetGeoAttnEncoder from schnet_geo_attn.py)
# ---------------------------------------------------------------------------

from torch_scatter import scatter_mean, scatter_sum


class SchNetGeoAttnV2Encoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3) -> None:
        super().__init__()

        self.num_gaussians = 50

        self.encoder = SchNetGeoAttnV2(
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
