#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
from torch_geometric.nn import knn_graph

from .schnet_geo_attn import GeoAttnInteractionBlock, SchNetGeoAttn


class SchNetUNetEncoderV22Semantic(nn.Module):
    """
    Semantic coarsening ablation of V19.

    Replaces FPS downsampling with predefined atom → residue/fragment
    semantic pooling.  All other architectural choices (GeoAttn backbone,
    gated skip decoder, interaction blocks, normalization, readout) follow
    V19 as closely as possible so that only the coarsening rule differs.

    Pipeline:
        atom-level encoder (n_atom_layers GeoAttn blocks)
            ↓  atom → residue attention pooling
        residue-level encoder (n_res_layers GeoAttn blocks)
            ↓  residue → atom gated unpooling  (V19-style skip gate)
        atom-level decoder (2 GeoAttn blocks)
            ↓  L2-normalise → scatter_sum → L2-normalise

    The atom → residue pooling uses:
        score_i  = atom2res_attn_mlp( [h_atom_i, b_embed[res_i], RBF(d_centroid_i)] )
        alpha_i  = softmax over atoms in the same residue
        msg_r    = Σ_i alpha_i * atom2res_value_lin(h_atom_i)
        h_res    = atom2res_norm( scatter_mean(h_atom, res) + sigmoid(pool_gate) * msg_r )

    The residue → atom gated unpooling uses (V19 skip-gate style):
        h_up     = h_res[orig_block_id]
        delta    = res2atom_delta_mlp( [h_atom_skip, h_up] )
        gate     = sigmoid( res2atom_gate_mlp( [h_atom_skip, h_up, RBF(d_centroid)] ) )
        h_atom   = h_atom_skip + gate * delta

    orig_block_id, batch_id_coarse, B_coarse must be provided by the caller
    (pretrain_model.py sets residue_pool=True to supply these).
    """

    def __init__(
        self,
        hidden_size,
        edge_size,
        n_layers=4,
        k_neighbors=9,
        gate_type='scalar',
        n_block_types=440,
        pool_gate_init=-2.0,
        skip_gate_init=-2.0,
    ):
        super().__init__()
        assert n_layers % 2 == 0, 'n_layers must be even (n_atom_layers = n_res_layers = n_layers // 2)'
        assert gate_type in ('scalar', 'channel'), \
            f"gate_type must be 'scalar' or 'channel', got '{gate_type}'"

        self.hidden_size    = hidden_size
        self.num_gaussians  = 32
        self.k_neighbors    = k_neighbors
        self.gate_type      = gate_type
        self.n_layers       = n_layers
        self.n_atom_layers  = n_layers // 2
        self.n_res_layers   = n_layers // 2

        # ── Backbone (identical to V19) ───────────────────────────────────
        self.schnet = SchNetGeoAttn(
            hidden_size,
            num_interactions=n_layers,
            num_gaussians=self.num_gaussians,
        )

        # ── Edge projection (identical to V19) ────────────────────────────
        self.edge_linear = (
            nn.Linear(edge_size, self.num_gaussians) if edge_size != 0 else None
        )

        # ── Decoder interaction blocks (identical to V19 style) ───────────
        # One unpooling step → 2 decoder blocks (same pattern as V19's n_skips*2)
        self.dec_interactions = nn.ModuleList([
            GeoAttnInteractionBlock(
                hidden_size, self.num_gaussians, hidden_size, self.schnet.cutoff
            )
            for _ in range(2)
        ])

        # ── Residue/block type embedding (from V8) ────────────────────────
        self.block_type_embedding = nn.Embedding(n_block_types, hidden_size)

        # ── Atom → residue attention pooling ─────────────────────────────
        # Input: [h_atom, b_embed[res_id], RBF(dist atom to residue centroid)]
        self.atom2res_attn_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + self.num_gaussians, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.atom2res_value_lin = nn.Linear(hidden_size, hidden_size)
        # Scalar gate: residual weight for pooled message, init ≈ 0.12
        self.atom2res_pool_gate = nn.Parameter(torch.tensor(pool_gate_init))
        self.atom2res_norm      = nn.LayerNorm(hidden_size)

        # ── Residue → atom gated unpooling (V19 skip-gate style) ─────────
        gate_out_dim = 1 if gate_type == 'scalar' else hidden_size
        self.res2atom_delta_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.res2atom_gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + self.num_gaussians, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, gate_out_dim),
        )
        # Conservative init: bias = skip_gate_init → sigmoid ≈ 0.12
        nn.init.constant_(self.res2atom_gate_mlp[-1].bias, skip_gate_init)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_edges(self, pos, batch):
        edge_index = knn_graph(pos, k=self.k_neighbors, batch=batch, loop=False)
        row, col   = edge_index
        dist       = (pos[row] - pos[col]).norm(dim=-1)
        rbf        = self.schnet.distance_expansion(dist)
        return edge_index, dist, rbf

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        H,
        Z,
        block_id,
        batch_id,
        edges,
        edge_attr=None,
        orig_block_id=None,
        batch_id_coarse=None,
        B_coarse=None,
    ):
        """
        H:               [N_atoms, hidden]    atom features (post BlockEmbedding)
        Z:               [N_atoms, 1, 3]      atom coordinates
        block_id:        [N_atoms]            identity mapping (atom_level=True)
        batch_id:        [N_atoms]            graph index per atom
        edges:           [2, E]               atom-level kNN edges
        edge_attr:       [E, edge_size]       edge type embeddings
        orig_block_id:   [N_atoms]            atom → residue/fragment index
        batch_id_coarse: [N_residues]         residue → graph index
        B_coarse:        [N_residues]         block type per residue/fragment
        """
        assert orig_block_id   is not None, 'orig_block_id must be provided'
        assert batch_id_coarse is not None, 'batch_id_coarse must be provided'
        assert B_coarse        is not None, 'B_coarse must be provided'

        H = scatter_mean(H, block_id, dim=0)
        Z = scatter_mean(Z, block_id, dim=0).squeeze(-2)  # [N_atoms, 3]

        assert orig_block_id.shape[0] == H.shape[0], (
            f'orig_block_id length ({orig_block_id.shape[0]}) != N_atoms ({H.shape[0]})'
        )
        assert batch_id_coarse.shape[0] == B_coarse.shape[0], (
            f'batch_id_coarse length ({batch_id_coarse.shape[0]}) != '
            f'B_coarse length ({B_coarse.shape[0]})'
        )

        Nc           = batch_id_coarse.shape[0]
        interactions = self.schnet.interactions

        # ── Atom-level edge RBF ───────────────────────────────────────────
        row, col = edges
        dist0    = (Z[row] - Z[col]).norm(dim=-1)
        rbf0     = self.schnet.distance_expansion(dist0)
        if edge_attr is not None and self.edge_linear is not None:
            rbf0 = rbf0 + self.edge_linear(edge_attr)

        # ── Atom-level encoder ────────────────────────────────────────────
        cur_h = H
        for i in range(self.n_atom_layers):
            cur_h = cur_h + interactions[i](cur_h, edges, dist0, rbf0)

        h_atom_skip = cur_h  # [N_atoms, hidden] — saved for gated unpooling

        # ── Atom → residue attention pooling ─────────────────────────────
        pos_res      = scatter_mean(Z, orig_block_id, dim=0, dim_size=Nc)  # [Nc, 3]
        h_res_init   = scatter_mean(cur_h, orig_block_id, dim=0, dim_size=Nc)  # [Nc, hidden]

        b_embed = self.block_type_embedding(B_coarse)   # [Nc, hidden]
        b_atom  = b_embed[orig_block_id]                # [N_atoms, hidden]

        dist_to_centroid = (Z - pos_res[orig_block_id]).norm(dim=-1)        # [N_atoms]
        rbf_centroid     = self.schnet.distance_expansion(dist_to_centroid) # [N_atoms, G]

        score = self.atom2res_attn_mlp(
            torch.cat([cur_h, b_atom, rbf_centroid], dim=-1)
        )                                                       # [N_atoms, 1]
        alpha = scatter_softmax(score, orig_block_id, dim=0)   # [N_atoms, 1]

        msg   = scatter_sum(
            alpha * self.atom2res_value_lin(cur_h),
            orig_block_id, dim=0, dim_size=Nc,
        )                                                       # [Nc, hidden]

        h_res = h_res_init + torch.sigmoid(self.atom2res_pool_gate) * msg
        h_res = self.atom2res_norm(h_res)                       # [Nc, hidden]

        # ── Residue-level edges ───────────────────────────────────────────
        res_edges, res_dist, res_rbf = self._build_edges(pos_res, batch_id_coarse)

        # ── Residue-level message passing ─────────────────────────────────
        for i in range(self.n_res_layers):
            h_res = h_res + interactions[self.n_atom_layers + i](
                h_res, res_edges, res_dist, res_rbf
            )

        # ── Residue → atom gated unpooling ────────────────────────────────
        h_res_to_atom = h_res[orig_block_id]  # [N_atoms, hidden]

        delta = self.res2atom_delta_mlp(
            torch.cat([h_atom_skip, h_res_to_atom], dim=-1)
        )                                                                    # [N_atoms, hidden]
        gate = torch.sigmoid(self.res2atom_gate_mlp(
            torch.cat([h_atom_skip, h_res_to_atom, rbf_centroid], dim=-1)
        ))                                                                   # [N_atoms, 1 or hidden]
        cur_h = h_atom_skip + gate * delta                                  # [N_atoms, hidden]

        # ── Atom-level decoder ────────────────────────────────────────────
        for i in range(2):
            cur_h = cur_h + self.dec_interactions[i](cur_h, edges, dist0, rbf0)

        # ── Output (identical to V19) ─────────────────────────────────────
        block_repr = F.normalize(cur_h, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)
        graph_repr = F.normalize(graph_repr, dim=-1)

        return H, block_repr, graph_repr, None
