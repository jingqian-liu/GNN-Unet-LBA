#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import fps, knn_graph
from torch_geometric.nn.pool import knn

from .schnet_geo_attn import GeoAttnInteractionBlock, SchNetGeoAttn


class SchNetUNetEncoderV10(nn.Module):
    """
    UNet-style SchNet encoder with GeoAttn message passing throughout.

    Identical to SchNetUNetEncoderV2 except both encoder and decoder
    interaction blocks use GeoAttnInteractionBlock instead of InteractionBlock.
    """

    def __init__(self, hidden_size, edge_size, n_layers=6,
                 fps_ratio=0.5, k_neighbors=9, interp_k=3):
        super().__init__()
        assert n_layers % 2 == 0, 'n_layers must be even (2 interaction blocks per level)'

        self.hidden_size   = hidden_size
        self.num_gaussians = 32
        self.fps_ratio     = fps_ratio
        self.k_neighbors   = k_neighbors
        self.interp_k      = interp_k
        self.n_levels      = n_layers // 2
        self.n_skips       = self.n_levels - 1

        # Encoder blocks — SchNetGeoAttn replaces SchNet
        self.schnet = SchNetGeoAttn(hidden_size, num_interactions=n_layers,
                                    num_gaussians=self.num_gaussians)

        self.edge_linear = (nn.Linear(edge_size, self.num_gaussians)
                            if edge_size != 0 else None)

        # Decoder blocks — GeoAttnInteractionBlock replaces InteractionBlock
        self.dec_interactions = nn.ModuleList([
            GeoAttnInteractionBlock(hidden_size, self.num_gaussians, hidden_size,
                                    self.schnet.cutoff)
            for _ in range(self.n_skips * 2)
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(2 * hidden_size, hidden_size) for _ in range(self.n_skips)
        ])

    def _build_edges(self, pos, batch):
        edge_index = knn_graph(pos, k=self.k_neighbors, batch=batch, loop=False)
        row, col   = edge_index
        dist       = (pos[row] - pos[col]).norm(dim=-1)
        rbf        = self.schnet.distance_expansion(dist)
        return edge_index, dist, rbf

    def _interpolate(self, h_coarse, pos_coarse, batch_coarse,
                     pos_fine, batch_fine):
        assign     = knn(pos_coarse, pos_fine, self.interp_k,
                         batch_x=batch_coarse, batch_y=batch_fine)
        fine_idx   = assign[0]
        coarse_idx = assign[1]

        dist   = (pos_fine[fine_idx] - pos_coarse[coarse_idx]).norm(dim=-1).clamp(min=1e-9)
        weight = 1.0 / dist
        weight = weight / scatter_sum(weight, fine_idx, dim=0,
                                      dim_size=pos_fine.shape[0])[fine_idx]

        return scatter_sum(
            weight.unsqueeze(-1) * h_coarse[coarse_idx],
            fine_idx, dim=0, dim_size=pos_fine.shape[0]
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H = scatter_mean(H, block_id, dim=0)
        Z = scatter_mean(Z, block_id, dim=0).squeeze(-2)  # [N, 3]

        enc_interactions = self.schnet.interactions

        row, col = edges
        dist0    = (Z[row] - Z[col]).norm(dim=-1)
        rbf0     = self.schnet.distance_expansion(dist0)
        if edge_attr is not None and self.edge_linear is not None:
            rbf0 = rbf0 + self.edge_linear(edge_attr)

        # ── Encoder ──────────────────────────────────────────────────────
        enc_h, enc_pos, enc_edges, enc_dist, enc_rbf, enc_batch = [], [], [], [], [], []
        cur_h     = H
        cur_pos   = Z
        cur_edges = edges
        cur_dist  = dist0
        cur_rbf   = rbf0
        cur_batch = batch_id

        for level in range(self.n_levels):
            base  = level * 2
            cur_h = cur_h + enc_interactions[base    ](cur_h, cur_edges, cur_dist, cur_rbf)
            cur_h = cur_h + enc_interactions[base + 1](cur_h, cur_edges, cur_dist, cur_rbf)

            if level < self.n_levels - 1:
                enc_h.append(cur_h)
                enc_pos.append(cur_pos)
                enc_edges.append(cur_edges)
                enc_dist.append(cur_dist)
                enc_rbf.append(cur_rbf)
                enc_batch.append(cur_batch)

                idx       = fps(cur_pos, cur_batch, ratio=self.fps_ratio)
                cur_h     = cur_h[idx]
                cur_pos   = cur_pos[idx]
                cur_batch = cur_batch[idx]
                cur_edges, cur_dist, cur_rbf = self._build_edges(cur_pos, cur_batch)

        # ── Decoder ──────────────────────────────────────────────────────
        for skip_i in range(self.n_skips - 1, -1, -1):
            fine_pos   = enc_pos[skip_i]
            fine_batch = enc_batch[skip_i]

            h_interp  = self._interpolate(cur_h, cur_pos, cur_batch,
                                          fine_pos, fine_batch)
            cur_h     = self.skip_lins[skip_i](torch.cat([h_interp, enc_h[skip_i]], dim=-1))
            cur_pos   = fine_pos
            cur_edges = enc_edges[skip_i]
            cur_dist  = enc_dist[skip_i]
            cur_rbf   = enc_rbf[skip_i]
            cur_batch = fine_batch

            dec_base = skip_i * 2
            cur_h = cur_h + self.dec_interactions[dec_base    ](cur_h, cur_edges, cur_dist, cur_rbf)
            cur_h = cur_h + self.dec_interactions[dec_base + 1](cur_h, cur_edges, cur_dist, cur_rbf)

        # ── Output ───────────────────────────────────────────────────────
        block_repr = F.normalize(cur_h, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        graph_repr = F.normalize(graph_repr, dim=-1)

        return H, block_repr, graph_repr, None
