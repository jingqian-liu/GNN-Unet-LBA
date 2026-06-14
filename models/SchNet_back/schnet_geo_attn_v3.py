#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
SchNet with pair-type-aware geometric attention on messages (v3).

Extends schnet_geo_attn.py by adding a pair-type embedding to the gate:

    e_pair  = pair_embedding(pair_type)          # pair_type ∈ {0=PP, 1=LL, 2=PL}
    a_ij    = σ( gate_mlp([h_i, h_j, RBF(d_ij), e_pair]) )
    m_ij    = a_ij · W(d_ij) · lin1(h_j)

Pair type is derived from segment_ids (0=protein, 1=ligand) passed by the
encoder wrapper. Designed specifically for protein-ligand tasks (LBA).
"""

from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing

from .schnet import GaussianSmearing, ShiftedSoftplus, SchNet


# ---------------------------------------------------------------------------
# Pair-type helpers
# ---------------------------------------------------------------------------

def compute_pair_type(segment_ids: Tensor, edge_index: Tensor) -> Tensor:
    """Return per-edge pair type: 0=PP, 1=LL, 2=PL.

    segment_ids: [N] per-atom, 0=protein, 1=ligand
    edge_index:  [2, E]
    """
    seg_i = segment_ids[edge_index[0]]   # [E]
    seg_j = segment_ids[edge_index[1]]   # [E]
    # PP → 0, LL → 1 (reuse segment value when equal), PL → 2
    return torch.where(seg_i == seg_j, seg_i.long(),
                       torch.full_like(seg_i, 2, dtype=torch.long))


# ---------------------------------------------------------------------------
# Conv / block
# ---------------------------------------------------------------------------

class PairAttnCFConv(MessagePassing):
    """CFConv with pair-type-aware geometric attention.

    Gate uses raw (pre-linear) node features + pair-type embedding:
        a_ij = σ( gate_mlp([h_i, h_j, rbf, e_pair]) )
        m_ij = a_ij · W(d_ij) · lin1(h_j)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_gaussians: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
        pair_embed_dim: int = 16,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn          # filter MLP: RBF -> num_filters (unchanged)
        self.cutoff = cutoff

        self.pair_embedding = Embedding(3, pair_embed_dim)   # 3 pair types

        self.gate_mlp = Sequential(
            Linear(2 * in_channels + num_gaussians + pair_embed_dim, num_filters),
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
                edge_attr: Tensor, pair_type: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)                # [E, num_filters]
        e_pair = self.pair_embedding(pair_type)                # [E, pair_embed_dim]

        # Pass raw x, filter W, rbf, and pair embedding to message
        out = self.propagate(edge_index, x=x, W=W, rbf=edge_attr, e_pair=e_pair)
        out = self.lin2(out)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, W: Tensor,
                rbf: Tensor, e_pair: Tensor) -> Tensor:
        a_ij = torch.sigmoid(
            self.gate_mlp(torch.cat([x_i, x_j, rbf, e_pair], dim=-1))
        )                                                      # [E, num_filters]
        return a_ij * W * self.lin1(x_j)                      # [E, num_filters]


class PairAttnInteractionBlock(torch.nn.Module):
    """Drop-in replacement for InteractionBlock using PairAttnCFConv."""

    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float, pair_embed_dim: int = 16):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = PairAttnCFConv(
            hidden_channels, hidden_channels, num_gaussians,
            num_filters, self.mlp, cutoff,
            pair_embed_dim=pair_embed_dim,
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
                edge_attr: Tensor, pair_type: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr, pair_type)
        x = self.act(x)
        x = self.lin(x)
        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class SchNetGeoAttnV3(SchNet):
    """SchNet with pair-type-aware geometric attention messages.

    forward() signature extended with pair_type [E] compared to SchNet.
    All other SchNet methods (distance_expansion, etc.) are inherited.
    """

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, pair_embed_dim: int = 16, **kwargs):
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
                PairAttnInteractionBlock(
                    hidden_channels, num_gaussians, num_filters, cutoff,
                    pair_embed_dim=pair_embed_dim,
                )
            )

    def forward(self, h: Tensor, pos: Tensor, batch: Tensor,
                edge_index: Tensor, edge_attr: Tensor,
                pair_type: Tensor) -> Tensor:
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        if edge_attr is not None:
            edge_attr = edge_attr + self.distance_expansion(edge_weight)
        else:
            edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr, pair_type)

        return h


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

from torch_scatter import scatter_mean, scatter_sum


class SchNetGeoAttnV3Encoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3,
                 pair_embed_dim: int = 16) -> None:
        super().__init__()

        self.num_gaussians = 50

        self.encoder = SchNetGeoAttnV3(
            hidden_size, num_interactions=n_layers,
            num_gaussians=self.num_gaussians,
            pair_embed_dim=pair_embed_dim,
        )

        self.edge_linear = (
            nn.Linear(edge_size, self.num_gaussians)
            if edge_size != 0 else None
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None,
                segment_ids=None):
        H = scatter_mean(H, block_id, dim=0)
        Z = scatter_mean(Z, block_id, dim=0).squeeze()
        if edge_attr is not None and self.edge_linear is not None:
            edge_attr = self.edge_linear(edge_attr)

        # Compute per-edge pair type from segment_ids
        pair_type = compute_pair_type(segment_ids, edges)    # [E]

        block_repr = self.encoder(H, Z, batch_id, edges, edge_attr, pair_type)
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None
