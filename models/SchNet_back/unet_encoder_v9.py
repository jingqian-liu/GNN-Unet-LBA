#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
from torch_geometric.nn import knn_graph

from .schnet import SchNet, InteractionBlock


class SchNetUNetEncoderV9(nn.Module):
    """
    2-level SchNet encoder without unpooling.

    Level 1 (atom):    n_atom_layers interaction blocks on the all-atom graph.
    Pooling:           attention pooling (atom → residue/fragment) using block-type embedding.
    Level 2 (residue): n_residue_layers interaction blocks on a kNN graph of residue centroids.
    Readout:           dual-branch graph pooling — no decoder.
                       g_atom  = atom_proj( scatter_sum(atom_h,   atom_batch) )
                       g_coarse = coarse_proj( scatter_sum(coarse_h, coarse_batch) )
                       g = fusion_mlp( cat([LayerNorm(g_atom), LayerNorm(g_coarse)]) )
                       returned as graph_repr; prediction head applies a final Linear.
    """

    def __init__(
        self,
        hidden_size,
        edge_size,
        n_atom_layers=3,
        n_residue_layers=2,
        k_neighbors=9,
        n_block_types=440,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_gaussians = 50
        self.k_neighbors = k_neighbors
        self.n_atom_layers = n_atom_layers
        self.n_residue_layers = n_residue_layers

        self.schnet = SchNet(
            hidden_size,
            num_interactions=n_atom_layers + n_residue_layers,
            num_gaussians=self.num_gaussians,
        )

        self.edge_linear = (
            nn.Linear(edge_size, self.num_gaussians)
            if edge_size != 0 else None
        )

        self.block_type_embedding = nn.Embedding(n_block_types, hidden_size)

        # Attention pooling
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        self.pool_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.pool_layernorm = nn.LayerNorm(hidden_size)

        # Dual-branch graph readout
        self.atom_proj = nn.Linear(hidden_size, hidden_size)
        self.coarse_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _build_edges(self, pos, batch):
        edge_index = knn_graph(pos, k=self.k_neighbors, batch=batch, loop=False)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        rbf = self.schnet.distance_expansion(dist)
        return edge_index, dist, rbf

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
        H:              [N_atoms, hidden]   atom features (post BlockEmbedding)
        Z:              [N_atoms, 1, 3]     atom coordinates
        block_id:       [N_atoms]           identity mapping (atom_level=True)
        batch_id:       [N_atoms]           graph index per atom
        edges:          [2, E]              atom-level kNN edges
        edge_attr:      [E, edge_size]      edge features
        orig_block_id:  [N_atoms]           atom -> residue/fragment index
        batch_id_coarse:[N_residues]        residue -> graph index
        B_coarse:       [N_residues]        block type index per residue/fragment
        """
        H = scatter_mean(H, block_id, dim=0)
        Z = scatter_mean(Z, block_id, dim=0).squeeze(-2)  # [N_atoms, 3]

        interactions = self.schnet.interactions

        # Atom-level RBF
        row, col = edges
        dist0 = (Z[row] - Z[col]).norm(dim=-1)
        rbf0 = self.schnet.distance_expansion(dist0)
        if edge_attr is not None and self.edge_linear is not None:
            rbf0 = rbf0 + self.edge_linear(edge_attr)

        # ── Level 1: atom-level message passing ──────────────────────────
        cur_h = H
        for i in range(self.n_atom_layers):
            cur_h = cur_h + interactions[i](cur_h, edges, dist0, rbf0)

        h_atom = cur_h  # [N_atoms, hidden]

        # ── Pooling: atom → residue/fragment ─────────────────────────────
        Nc = batch_id_coarse.shape[0]

        pos_coarse = scatter_mean(Z, orig_block_id, dim=0, dim_size=Nc)      # [Nc, 3]

        b_embed = self.block_type_embedding(B_coarse)                         # [Nc, hidden]
        b_atom = b_embed[orig_block_id]                                       # [N_atoms, hidden]

        score = self.gate_mlp(torch.cat([cur_h, b_atom], dim=-1))            # [N_atoms, 1]
        alpha = scatter_softmax(score, orig_block_id, dim=0)                  # [N_atoms, 1]

        value = self.value_proj(cur_h)                                        # [N_atoms, hidden]
        h_pool = scatter_sum(alpha * value, orig_block_id, dim=0, dim_size=Nc)  # [Nc, hidden]

        h_coarse = self.pool_layernorm(
            self.pool_mlp(torch.cat([h_pool, b_embed], dim=-1))
        )                                                                      # [Nc, hidden]

        # ── Level 2: residue-level message passing ────────────────────────
        coarse_edges, coarse_dist, coarse_rbf = self._build_edges(
            pos_coarse, batch_id_coarse
        )
        for i in range(self.n_residue_layers):
            h_coarse = h_coarse + interactions[self.n_atom_layers + i](
                h_coarse, coarse_edges, coarse_dist, coarse_rbf
            )

        # ── Dual-branch graph readout (no unpooling) ──────────────────────
        batch_size = batch_id.max().item() + 1

        g_atom = self.atom_proj(
            scatter_sum(h_atom, batch_id, dim=0, dim_size=batch_size)
        )                                                                      # [bs, hidden]
        g_coarse = self.coarse_proj(
            scatter_sum(h_coarse, batch_id_coarse, dim=0, dim_size=batch_size)
        )                                                                      # [bs, hidden]

        g_atom = F.layer_norm(g_atom, g_atom.shape[-1:])
        g_coarse = F.layer_norm(g_coarse, g_coarse.shape[-1:])

        graph_repr = self.fusion_mlp(torch.cat([g_atom, g_coarse], dim=-1))  # [bs, hidden]

        # block_repr returned for API compatibility (not used by graph_level_pred head)
        block_repr = F.normalize(h_atom, dim=-1)                             # [N_atoms, hidden]

        return H, block_repr, graph_repr, None
