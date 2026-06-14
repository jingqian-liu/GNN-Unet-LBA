#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from .modules.get import GET


class GETEncoder(nn.Module):
    def __init__(self, hidden_size, radial_size, n_channel,
                 n_rbf=1, cutoff=7.0, edge_size=16, n_layers=3,
                 n_head=1, dropout=0.1,
                 z_requires_grad=True, stable=False) -> None:
        super().__init__()

        self.encoder = GET(
            hidden_size, radial_size, n_channel,
            n_rbf, cutoff, edge_size, n_layers,
            n_head, dropout=dropout,
            z_requires_grad=z_requires_grad
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        # if not getattr(self, '_dims_printed', False):
        #     Nb = batch_id.shape[0]
        #     Eb = edges.shape[1]
        #     print('\n========== GET Feature Dimensions ==========', file=sys.stderr)
        #     print(f'[Unit level]  N_atoms={H.shape[0]},  H={tuple(H.shape)},  Z={tuple(Z.shape)}', file=sys.stderr)
        #     print(f'[Block level] N_blocks={Nb},  block edges E_b={Eb}', end='', file=sys.stderr)
        #     print(f',  edge_attr={tuple(edge_attr.shape)}' if edge_attr is not None else ',  edge_attr=None', file=sys.stderr)
        #     self._dims_printed = True

        H, pred_Z = self.encoder(H, Z, block_id, batch_id, edges, edge_attr)

        # if not getattr(self, '_dims_printed2', False):
        #     block_repr_raw = scatter_sum(H, block_id, dim=0)
        #     print(f'[After GET]   H={tuple(H.shape)},  block_repr (pre-norm)={tuple(block_repr_raw.shape)}', file=sys.stderr)
        #     self._dims_printed2 = True

        # block_repr = scatter_mean(H, block_id, dim=0)           # [Nb, hidden]
        block_repr = scatter_sum(H, block_id, dim=0)           # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        # graph_repr = scatter_mean(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)

        # if not getattr(self, '_dims_printed3', False):
        #     print(f'[Readout]     block_repr={tuple(block_repr.shape)},  graph_repr={tuple(graph_repr.shape)}', file=sys.stderr)
        #     print('============================================\n', file=sys.stderr)
        #     self._dims_printed3 = True

        return H, block_repr, graph_repr, pred_Z