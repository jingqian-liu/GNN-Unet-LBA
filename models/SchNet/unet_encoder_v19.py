#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
from torch_geometric.nn import fps, knn_graph
from torch_geometric.nn.pool import knn

from .schnet_geo_attn import GeoAttnInteractionBlock, SchNetGeoAttn


class SchNetUNetEncoderV19(nn.Module):
    """
    V14 + geometry-aware attention pooling at each FPS downsampling step.

    FPS still selects coarse centers (geometric coverage unchanged).
    After selection, each center aggregates a small residual message from
    its pool_k nearest fine-level neighbors via attention pooling:

        assign      = kNN(pos_fine → pos_center, k=pool_k)
        score_cj    = pool_attn_mlp([h_center, h_fine, RBF(d_cj)])
        alpha_cj    = softmax over j for each center c
        msg_c       = Σ_j alpha_cj * value_lin(h_fine_j)
        h_coarse    = h_center + sigmoid(pool_gate) * msg_c

    pool_gate is a per-level learnable scalar initialised to -2 so that
    sigmoid(-2) ≈ 0.12: the model starts near V14 (copy of FPS center
    feature) and gradually opens the pooling if it is beneficial.

    Each downsampling level has its own pool_attn_mlp, pool_value_lin,
    and pool_gate (separate weights, n_skips sets total).

    All other design choices (GeoAttn backbone, conservative skip-gate init)
    are identical to V14.
    """

    def __init__(self, hidden_size, edge_size, n_layers=6,
                 fps_ratio=0.5, k_neighbors=9, interp_k=3, pool_k=9,
                 gate_type='scalar'):
        super().__init__()
        assert n_layers % 2 == 0, 'n_layers must be even (2 interaction blocks per level)'
        assert gate_type in ('scalar', 'channel'), \
            f"gate_type must be 'scalar' or 'channel', got '{gate_type}'"

        self.hidden_size   = hidden_size
        self.num_gaussians = 32
        self.fps_ratio     = fps_ratio
        self.k_neighbors   = k_neighbors
        self.interp_k      = interp_k
        self.pool_k        = pool_k
        self.n_levels      = n_layers // 2
        self.n_skips       = self.n_levels - 1
        self.gate_type     = gate_type

        self.schnet = SchNetGeoAttn(hidden_size, num_interactions=n_layers,
                                    num_gaussians=self.num_gaussians)

        self.edge_linear = (nn.Linear(edge_size, self.num_gaussians)
                            if edge_size != 0 else None)

        self.dec_interactions = nn.ModuleList([
            GeoAttnInteractionBlock(hidden_size, self.num_gaussians, hidden_size,
                                    self.schnet.cutoff)
            for _ in range(self.n_skips * 2)
        ])

        # ── Skip-connection gate (identical to V14) ───────────────────────
        gate_out_dim = 1 if gate_type == 'scalar' else hidden_size
        self.skip_delta_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(self.n_skips)
        ])
        self.skip_gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_size + self.num_gaussians, hidden_size // 2),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, gate_out_dim),
            )
            for _ in range(self.n_skips)
        ])
        # Conservative skip-gate init: bias = -2 → initial gate ≈ 0.12
        for mlp in self.skip_gate_mlps:
            nn.init.constant_(mlp[-1].bias, -2.0)

        # ── Attention pooling at downsampling — one set per level ─────────
        # pool_attn_mlp: [h_center, h_fine, rbf] → attention logit
        self.pool_attn_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_size + self.num_gaussians, hidden_size // 2),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, 1),
            )
            for _ in range(self.n_skips)
        ])
        # pool_value_lin: projects fine-node features before aggregation
        self.pool_value_lins = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(self.n_skips)
        ])
        # pool_gate: scalar residual weight per level, init = -2 → ≈ 0.12
        self.pool_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(-2.0))
            for _ in range(self.n_skips)
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_edges(self, pos, batch):
        edge_index = knn_graph(pos, k=self.k_neighbors, batch=batch, loop=False)
        row, col   = edge_index
        dist       = (pos[row] - pos[col]).norm(dim=-1)
        rbf        = self.schnet.distance_expansion(dist)
        return edge_index, dist, rbf

    def _interpolate(self, h_coarse, pos_coarse, batch_coarse,
                     pos_fine, batch_fine):
        """Returns (h_interp, avg_dist) — avg_dist used as geometry input to gate."""
        assign     = knn(pos_coarse, pos_fine, self.interp_k,
                         batch_x=batch_coarse, batch_y=batch_fine)
        fine_idx   = assign[0]
        coarse_idx = assign[1]

        dist        = (pos_fine[fine_idx] - pos_coarse[coarse_idx]).norm(dim=-1).clamp(min=1e-9)
        weight      = 1.0 / dist
        weight_sum  = scatter_sum(weight, fine_idx, dim=0, dim_size=pos_fine.shape[0])
        weight_norm = weight / weight_sum[fine_idx]

        h_interp = scatter_sum(
            weight_norm.unsqueeze(-1) * h_coarse[coarse_idx],
            fine_idx, dim=0, dim_size=pos_fine.shape[0],
        )
        avg_dist = scatter_sum(
            weight_norm * dist, fine_idx, dim=0, dim_size=pos_fine.shape[0],
        )
        return h_interp, avg_dist

    def _fps_with_attn_pool(self, h, pos, batch, level_idx):
        """
        FPS downsampling followed by geometry-aware attention pooling.

        Returns (idx, h_coarse):
          idx      — FPS-selected global node indices (used to update pos/batch)
          h_coarse — coarse features: h_center + sigmoid(pool_gate) * pooled_msg
        """
        # ── 1. FPS selection ──────────────────────────────────────────────
        idx          = fps(pos, batch, ratio=self.fps_ratio, random_start=False)
        h_center     = h[idx]                   # [N_c, hidden]
        pos_center   = pos[idx]                  # [N_c, 3]
        batch_center = batch[idx]                # [N_c]
        N_c          = idx.numel()

        # ── 2. kNN: for each coarse center find pool_k nearest fine nodes ─
        assign     = knn(pos, pos_center, self.pool_k,
                         batch_x=batch, batch_y=batch_center)
        center_idx = assign[0]   # [E_pool] — indexes into coarse centers
        fine_idx   = assign[1]   # [E_pool] — indexes into fine nodes

        dist = (pos_center[center_idx] - pos[fine_idx]).norm(dim=-1)  # [E_pool]
        rbf  = self.schnet.distance_expansion(dist)                    # [E_pool, num_gaussians]

        # ── 3. Attention scores (softmax-normalised per center) ───────────
        attn_input = torch.cat(
            [h_center[center_idx], h[fine_idx], rbf], dim=-1
        )                                                               # [E_pool, 2h+rbf]
        score = self.pool_attn_mlps[level_idx](attn_input)             # [E_pool, 1]
        alpha = scatter_softmax(score, center_idx, dim=0)               # [E_pool, 1]

        # ── 4. Weighted value aggregation ─────────────────────────────────
        value = self.pool_value_lins[level_idx](h[fine_idx])           # [E_pool, hidden]
        msg   = scatter_sum(
            alpha * value, center_idx, dim=0, dim_size=N_c
        )                                                               # [N_c, hidden]

        # ── 5. Conservative residual update ───────────────────────────────
        gate     = torch.sigmoid(self.pool_gates[level_idx])           # scalar ≈ 0.12
        h_coarse = h_center + gate * msg

        return idx, h_coarse

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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

                # FPS + attention pooling (replaces plain FPS copy)
                idx, cur_h = self._fps_with_attn_pool(cur_h, cur_pos, cur_batch, level)
                cur_pos    = cur_pos[idx]
                cur_batch  = cur_batch[idx]
                cur_edges, cur_dist, cur_rbf = self._build_edges(cur_pos, cur_batch)

        # ── Decoder ──────────────────────────────────────────────────────
        for skip_i in range(self.n_skips - 1, -1, -1):
            fine_pos   = enc_pos[skip_i]
            fine_batch = enc_batch[skip_i]

            h_interp, avg_dist = self._interpolate(
                cur_h, cur_pos, cur_batch, fine_pos, fine_batch,
            )

            h_skip     = enc_h[skip_i]
            rbf_interp = self.schnet.distance_expansion(avg_dist)

            delta = self.skip_delta_mlps[skip_i](
                torch.cat([h_skip, h_interp], dim=-1)
            )
            gate = torch.sigmoid(self.skip_gate_mlps[skip_i](
                torch.cat([h_skip, h_interp, rbf_interp], dim=-1)
            ))
            cur_h = h_skip + gate * delta

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
