#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax, scatter_max
from torch_geometric.nn import fps, knn_graph
from torch_geometric.nn.pool import knn

from .schnet import SchNet, InteractionBlock

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_sum, scatter_mean, scatter_softmax, scatter_max
from torch_geometric.nn import fps, knn_graph
from torch_geometric.nn.pool import knn


class SchNetUNetEncoderV6(nn.Module):
    """
    SchNet-U-Net with Point Pooling.

    Main changes from V9:
        1. Replace FPS-kNN soft pooling with Point Pooling.
        2. FPS selects coarse centers.
        3. Each center pools its original 1-hop neighbors.
        4. Clusters can overlap.
        5. Keep normalized-sum graph readout.
    """

    def __init__(
        self,
        hidden_size,
        edge_size,
        n_layers=6,
        n_levels=2,
        fps_ratio=0.6,
        k_neighbors=9,
        interp_k=3,
        n_decoder_blocks=1,
    ):
        super().__init__()

        assert n_layers % n_levels == 0, "n_layers must be divisible by n_levels"

        self.hidden_size = hidden_size
        self.num_gaussians = 50
        self.fps_ratio = fps_ratio
        self.k_neighbors = k_neighbors
        self.interp_k = interp_k
        self.n_levels = n_levels
        self.n_skips = n_levels - 1
        self.n_blocks_per_level = n_layers // n_levels
        self.n_decoder_blocks = n_decoder_blocks

        self.schnet = SchNet(
            hidden_size,
            num_interactions=n_layers,
            num_gaussians=self.num_gaussians,
        )

        self.edge_linear = (
            nn.Linear(edge_size, self.num_gaussians)
            if edge_size != 0 else None
        )

        self.dec_interactions = nn.ModuleList([
            InteractionBlock(
                hidden_size,
                self.num_gaussians,
                hidden_size,
                self.schnet.cutoff,
            )
            for _ in range(self.n_skips * n_decoder_blocks)
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(2 * hidden_size, hidden_size)
            for _ in range(self.n_skips)
        ])

        # Point-pooling attention score.
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

        # Fuse sum/max/attention pooled features.
        self.pool_fuse = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _build_edges(self, pos, batch):
        edge_index = knn_graph(
            pos,
            k=self.k_neighbors,
            batch=batch,
            loop=False,
        )
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        rbf = self.schnet.distance_expansion(dist)
        return edge_index, dist, rbf

    def _point_pool(self, h_fine, pos_fine, batch_fine, edge_index_fine):
        """
        Point Pooling:
            1. FPS selects center nodes.
            2. Each center pools its 1-hop neighbors in the original fine graph.
            3. A fine node can be 1-hop neighbor of multiple centers,
               so clusters can overlap.

        h_fine: [Nf, hidden]
        pos_fine: [Nf, 3]
        batch_fine: [Nf]
        edge_index_fine: [2, Ef]

        returns:
            h_coarse: [Nc, hidden]
            pos_coarse: [Nc, 3]
            batch_coarse: [Nc]
        """

        center_idx = fps(pos_fine, batch_fine, ratio=self.fps_ratio)

        pos_coarse = pos_fine[center_idx]
        batch_coarse = batch_fine[center_idx]

        Nf = h_fine.size(0)
        Nc = center_idx.size(0)

        center_map = torch.full(
            (Nf,),
            -1,
            device=h_fine.device,
            dtype=torch.long,
        )
        center_map[center_idx] = torch.arange(Nc, device=h_fine.device)

        row, col = edge_index_fine

        # Case 1: center -> neighbor
        mask_row_center = center_map[row] >= 0
        coarse_from_row = center_map[row[mask_row_center]]
        fine_from_col = col[mask_row_center]

        # Case 2: neighbor -> center
        mask_col_center = center_map[col] >= 0
        coarse_from_col = center_map[col[mask_col_center]]
        fine_from_row = row[mask_col_center]

        # Self-loop: center pools itself.
        self_coarse = torch.arange(Nc, device=h_fine.device)
        self_fine = center_idx

        coarse_idx = torch.cat(
            [coarse_from_row, coarse_from_col, self_coarse],
            dim=0,
        )
        fine_idx = torch.cat(
            [fine_from_col, fine_from_row, self_fine],
            dim=0,
        )

        # Remove duplicate (coarse, fine) pairs.
        pair = coarse_idx * Nf + fine_idx
        pair_unique = torch.unique(pair)
        coarse_idx = pair_unique // Nf
        fine_idx = pair_unique % Nf

        # Attention pooling within each point-pooling cluster.
        score = self.pool_score(h_fine[fine_idx])          # [P, 1]
        alpha = scatter_softmax(score, coarse_idx, dim=0)  # [P, 1]

        h_attn = scatter_sum(
            alpha * h_fine[fine_idx],
            coarse_idx,
            dim=0,
            dim_size=Nc,
        )

        h_sum = scatter_sum(
            h_fine[fine_idx],
            coarse_idx,
            dim=0,
            dim_size=Nc,
        )

        h_max = scatter_max(
            h_fine[fine_idx],
            coarse_idx,
            dim=0,
            dim_size=Nc,
        )[0]

        # Safety: scatter_max can return -inf if a group is empty.
        # It should not happen because each center has self-loop, but this is safe.
        h_max = torch.where(torch.isfinite(h_max), h_max, torch.zeros_like(h_max))

        h_coarse = self.pool_fuse(
            torch.cat([h_sum, h_max, h_attn], dim=-1)
        )

        return h_coarse, pos_coarse, batch_coarse

    def _interpolate(self, h_coarse, pos_coarse, batch_coarse, pos_fine, batch_fine):
        """
        kNN inverse-distance interpolation:
            coarse features -> fine nodes
        """
        assign = knn(
            pos_coarse,
            pos_fine,
            self.interp_k,
            batch_x=batch_coarse,
            batch_y=batch_fine,
        )

        fine_idx = assign[0]
        coarse_idx = assign[1]

        dist = (
            pos_fine[fine_idx] - pos_coarse[coarse_idx]
        ).norm(dim=-1).clamp(min=1e-9)

        weight = 1.0 / dist
        weight = weight / scatter_sum(
            weight,
            fine_idx,
            dim=0,
            dim_size=pos_fine.shape[0],
        )[fine_idx].clamp(min=1e-9)

        h_fine = scatter_sum(
            weight.unsqueeze(-1) * h_coarse[coarse_idx],
            fine_idx,
            dim=0,
            dim_size=pos_fine.shape[0],
        )

        return h_fine

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        """
        H: [num_units, hidden_size]
        Z: [num_units, n_channel, 3] or [num_units, 1, 3]
        block_id: [num_units], maps unit/atom to block/node
        batch_id: [num_blocks]
        edges: [2, E], full-resolution graph
        edge_attr: [E, edge_size]
        """

        # Unit/atom -> block/node initial representation.
        H = scatter_mean(H, block_id, dim=0)              # [N, hidden]
        Z = scatter_mean(Z, block_id, dim=0).squeeze(-2)  # [N, 3]

        enc_interactions = self.schnet.interactions
        bpl = self.n_blocks_per_level

        row, col = edges
        dist0 = (Z[row] - Z[col]).norm(dim=-1)
        rbf0 = self.schnet.distance_expansion(dist0)

        if edge_attr is not None and self.edge_linear is not None:
            rbf0 = rbf0 + self.edge_linear(edge_attr)

        enc_h, enc_pos, enc_edges, enc_dist, enc_rbf, enc_batch = [], [], [], [], [], []

        cur_h = H
        cur_pos = Z
        cur_edges = edges
        cur_dist = dist0
        cur_rbf = rbf0
        cur_batch = batch_id

        # -------------------------
        # Encoder
        # -------------------------
        for level in range(self.n_levels):
            base = level * bpl

            for b in range(bpl):
                cur_h = cur_h + enc_interactions[base + b](
                    cur_h,
                    cur_edges,
                    cur_dist,
                    cur_rbf,
                )

            if level < self.n_levels - 1:
                enc_h.append(cur_h)
                enc_pos.append(cur_pos)
                enc_edges.append(cur_edges)
                enc_dist.append(cur_dist)
                enc_rbf.append(cur_rbf)
                enc_batch.append(cur_batch)

                cur_h, cur_pos, cur_batch = self._point_pool(
                    cur_h,
                    cur_pos,
                    cur_batch,
                    cur_edges,
                )

                cur_edges, cur_dist, cur_rbf = self._build_edges(
                    cur_pos,
                    cur_batch,
                )

        # -------------------------
        # Decoder
        # -------------------------
        for skip_i in range(self.n_skips - 1, -1, -1):
            fine_pos = enc_pos[skip_i]
            fine_batch = enc_batch[skip_i]

            h_interp = self._interpolate(
                cur_h,
                cur_pos,
                cur_batch,
                fine_pos,
                fine_batch,
            )

            cur_h = self.skip_lins[skip_i](
                torch.cat([h_interp, enc_h[skip_i]], dim=-1)
            )

            cur_pos = fine_pos
            cur_edges = enc_edges[skip_i]
            cur_dist = enc_dist[skip_i]
            cur_rbf = enc_rbf[skip_i]
            cur_batch = fine_batch

            dec_base = skip_i * self.n_decoder_blocks
            for b in range(self.n_decoder_blocks):
                cur_h = cur_h + self.dec_interactions[dec_base + b](
                    cur_h,
                    cur_edges,
                    cur_dist,
                    cur_rbf,
                )

        # -------------------------
        # Readout
        # -------------------------
        block_repr = F.normalize(cur_h, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        graph_repr = F.normalize(graph_repr, dim=-1)

        return H, block_repr, graph_repr, None
