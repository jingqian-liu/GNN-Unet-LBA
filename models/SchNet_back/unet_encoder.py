#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import fps, knn_graph
from torch_geometric.nn.pool import knn

from .schnet import SchNet


class SchNetUNetEncoder(nn.Module):
    """
    UNet-style SchNet encoder.

    Encoder: n_levels resolution levels, each with 2 interaction blocks.
             FPS (ratio=fps_ratio) downsamples between levels.
    Decoder: mirrors the encoder using the same interaction block weights (weight tying).
             kNN interpolation (weighted by 1/dist) upsamples back to each finer level.
             Skip connections concatenate encoder features then project back to hidden_size.

    With n_layers=6: 3 levels (full → N/2 → N/4), 2 skips.
    """

    def __init__(self, hidden_size, edge_size, n_layers=6,
                 fps_ratio=0.5, k_neighbors=9, interp_k=3):
        super().__init__()
        assert n_layers % 2 == 0, 'n_layers must be even (2 interaction blocks per level)'

        self.hidden_size   = hidden_size
        self.num_gaussians = 50
        self.fps_ratio     = fps_ratio
        self.k_neighbors   = k_neighbors
        self.interp_k      = interp_k
        self.n_levels      = n_layers // 2       # 3 for n_layers=6
        self.n_skips       = self.n_levels - 1   # 2 for n_levels=3

        self.schnet = SchNet(hidden_size, num_interactions=n_layers,
                             num_gaussians=self.num_gaussians)

        self.edge_linear = (nn.Linear(edge_size, self.num_gaussians)
                            if edge_size != 0 else None)

        # One Linear(2h → h) per upsample step
        self.skip_lins = nn.ModuleList([
            nn.Linear(2 * hidden_size, hidden_size) for _ in range(self.n_skips)
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_edges(self, pos, batch):
        """kNN graph + distances + RBF for a (possibly subsampled) point cloud."""
        edge_index = knn_graph(pos, k=self.k_neighbors, batch=batch, loop=False)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        rbf  = self.schnet.distance_expansion(dist)
        return edge_index, dist, rbf

    def _interpolate(self, h_coarse, pos_coarse, batch_coarse,
                     pos_fine, batch_fine):
        """Inverse-distance weighted interpolation: coarse → fine."""
        assign     = knn(pos_coarse, pos_fine, self.interp_k,
                         batch_x=batch_coarse, batch_y=batch_fine)
        fine_idx   = assign[0]   # indices into pos_fine
        coarse_idx = assign[1]   # indices into pos_coarse

        dist   = (pos_fine[fine_idx] - pos_coarse[coarse_idx]).norm(dim=-1).clamp(min=1e-9)
        weight = 1.0 / dist
        weight = weight / scatter_sum(weight, fine_idx, dim=0,
                                      dim_size=pos_fine.shape[0])[fine_idx]

        return scatter_sum(
            weight.unsqueeze(-1) * h_coarse[coarse_idx],
            fine_idx, dim=0, dim_size=pos_fine.shape[0]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H = scatter_mean(H, block_id, dim=0)            # [N, h]
        Z = scatter_mean(Z, block_id, dim=0).squeeze(-2)  # [N, 3]

        interactions = self.schnet.interactions

        # Full-resolution RBF (edges pre-built by KNNBatchEdgeConstructor)
        row, col = edges
        dist0 = (Z[row] - Z[col]).norm(dim=-1)
        rbf0  = self.schnet.distance_expansion(dist0)
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
            base    = level * 2
            cur_h   = cur_h + interactions[base    ](cur_h, cur_edges, cur_dist, cur_rbf)
            cur_h   = cur_h + interactions[base + 1](cur_h, cur_edges, cur_dist, cur_rbf)

            if level < self.n_levels - 1:   # downsample (skip at bottleneck)
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

            # Reuse encoder blocks for this level (reversed order within the pair)
            base    = skip_i * 2
            cur_h   = cur_h + interactions[base + 1](cur_h, cur_edges, cur_dist, cur_rbf)
            cur_h   = cur_h + interactions[base    ](cur_h, cur_edges, cur_dist, cur_rbf)

        # ── Output ───────────────────────────────────────────────────────
        block_repr = F.normalize(cur_h, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        graph_repr = F.normalize(graph_repr, dim=-1)

        return H, block_repr, graph_repr, None
