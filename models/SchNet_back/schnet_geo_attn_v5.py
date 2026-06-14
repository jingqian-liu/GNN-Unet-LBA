#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
SchNet with local edge softmax attention (v5).

Based on schnet_geo_attn.py. No residual connection, no edge-type embedding.

Replaces the sigmoid gate with proper softmax-normalized attention over the
neighborhood of each target node:

    q_i  = q_proj(h_i)                       # [E, d_attn]
    k_j  = k_proj(h_j)                       # [E, d_attn]

    content_score = (q_i · k_j) / √d_attn    # [E, 1]  scalar
    geom_score    = geom_bias_mlp(rbf)        # [E, 1]  distance bias
    a_ij          = content_score + geom_score

    α_ij = softmax_{j ∈ N(i)}(a_ij)          # [E, 1]  normalized per target

    W    = filter_mlp(rbf) * cosine_cutoff    # [E, F]  SchNet filter (unchanged)
    v    = value_proj(h_j)                    # [E, F]
    m_i  = Σ_j  α_ij · W · v                 # aggregated to target i

α_ij is scalar — same weight applied to every channel of the message.
Aggregation is done manually with scatter_sum (no MessagePassing base class
needed because scatter_softmax already sees all edges of each node at once).
"""

import math
from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential

from torch_scatter import scatter_softmax, scatter_sum

from .schnet import ShiftedSoftplus, SchNet


class LocalEdgeAttnCFConv(nn.Module):
    """CFConv with local softmax attention.

    Scalar attention weight α_ij is normalized over the neighbors j of each
    target node i via scatter_softmax, then broadcast over filter channels.

    Requires edge_index convention: edge_index[0]=source(j), edge_index[1]=target(i).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_gaussians: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
        d_attn: int = 16,
    ):
        super().__init__()
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn          # filter MLP: RBF -> num_filters (unchanged)
        self.cutoff = cutoff
        self.d_attn = d_attn

        # QK projections for content score (scalar dot-product attention)
        self.q_proj = Linear(in_channels, d_attn, bias=False)
        self.k_proj = Linear(in_channels, d_attn, bias=False)

        # Geometry bias: RBF -> scalar, adds distance-awareness to the score
        self.geom_bias_mlp = Sequential(
            Linear(num_gaussians, num_gaussians // 2),
            ShiftedSoftplus(),
            Linear(num_gaussians // 2, 1),
        )

        # Value projection (source features projected into filter space)
        self.value_proj = Linear(in_channels, num_filters, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.geom_bias_mlp[0].weight)
        self.geom_bias_mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.geom_bias_mlp[2].weight)
        self.geom_bias_mlp[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.value_proj.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        src, dst = edge_index          # src=j (source), dst=i (target)
        N = x.shape[0]

        h_i = x[dst]                  # [E, in_channels]
        h_j = x[src]                  # [E, in_channels]

        # ── Attention score ──────────────────────────────────────────────
        q = self.q_proj(h_i)          # [E, d_attn]
        k = self.k_proj(h_j)          # [E, d_attn]

        content_score = (q * k).sum(-1, keepdim=True) / math.sqrt(self.d_attn)
        geom_score = self.geom_bias_mlp(edge_attr)     # [E, 1]

        score = content_score + geom_score             # [E, 1]

        # Softmax over all source neighbors j for each target i
        alpha = scatter_softmax(score, dst, dim=0)     # [E, 1]

        # ── Message ──────────────────────────────────────────────────────
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)         # [E, num_filters]

        v = self.value_proj(h_j)                       # [E, num_filters]
        msg = alpha * W * v                            # [E, num_filters]

        # ── Aggregation ──────────────────────────────────────────────────
        out = scatter_sum(msg, dst, dim=0, dim_size=N) # [N, num_filters]
        out = self.lin2(out)                           # [N, out_channels]
        return out


class LocalEdgeAttnInteractionBlock(torch.nn.Module):
    """Drop-in replacement for InteractionBlock using LocalEdgeAttnCFConv."""

    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float, d_attn: int = 16):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = LocalEdgeAttnCFConv(
            hidden_channels, hidden_channels, num_gaussians,
            num_filters, self.mlp, cutoff, d_attn=d_attn,
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


class SchNetGeoAttnV5(SchNet):
    """SchNet with local softmax edge attention.

    Identical to SchNet except interactions use LocalEdgeAttnInteractionBlock.
    Inherits distance_expansion and all other methods unchanged.
    Overrides forward() only to pass edge_index directly (no edge_weight recompute).
    """

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, d_attn: int = 16, **kwargs):
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
                LocalEdgeAttnInteractionBlock(
                    hidden_channels, num_gaussians, num_filters, cutoff,
                    d_attn=d_attn,
                )
            )


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

from torch_scatter import scatter_mean


class SchNetGeoAttnV5Encoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3,
                 d_attn: int = None) -> None:
        super().__init__()

        self.num_gaussians = 50
        if d_attn is None:
            d_attn = max(hidden_size // 4, 8)   # e.g. 16 when hidden=64

        # num_filters = hidden_size to keep dimensions consistent
        self.encoder = SchNetGeoAttnV5(
            hidden_size,
            num_filters=hidden_size,
            num_interactions=n_layers,
            num_gaussians=self.num_gaussians,
            d_attn=d_attn,
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
