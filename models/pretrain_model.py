#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple
from copy import deepcopy
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum

from data.pdb_utils import VOCAB

from .GET.modules.tools import BlockEmbedding, KNNBatchEdgeConstructor


ReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 'noise', 'noise_level',
     'unit_repr', 'block_repr', 'graph_repr',
     'batch_id', 'block_id',
     'loss', 'noise_loss', 'noise_level_loss', 'align_loss'],
    )


def construct_edges(edge_constructor, B, batch_id, segment_ids, X, block_id, complexity=-1):
    if complexity == -1:  # don't do splicing
        intra_edges, inter_edges, global_global_edges, global_normal_edges, _ = edge_constructor(B, batch_id, segment_ids, X=X, block_id=block_id)
        return intra_edges, inter_edges, global_global_edges, global_normal_edges

    # do splicing
    offset, bs_id_start, bs_id_end = 0, 0, 0
    mini_intra_edges, mini_inter_edges, mini_global_global_edges, mini_global_normal_edges = [], [], [], []
    with torch.no_grad():
        batch_size = batch_id.max() + 1
        unit_batch_id = batch_id[block_id]
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id, dim=0)

        while bs_id_end < batch_size:
            bs_id_start = bs_id_end
            bs_id_end += 1
            while bs_id_end + 1 <= batch_size and \
                  (lengths[bs_id_start:bs_id_end + 1] * lengths[bs_id_start:bs_id_end + 1].max()).sum() < complexity:
                bs_id_end += 1
            # print(bs_id_start, bs_id_end, lengths[bs_id_start:bs_id_end], (lengths[bs_id_start:bs_id_end] * lengths[bs_id_start:bs_id_end].max()).sum())
            
            block_is_in = (batch_id >= bs_id_start) & (batch_id < bs_id_end)
            unit_is_in = (unit_batch_id >= bs_id_start) & (unit_batch_id < bs_id_end)
            B_mini, batch_id_mini, segment_ids_mini = B[block_is_in], batch_id[block_is_in], segment_ids[block_is_in]
            X_mini, block_id_mini = X[unit_is_in], block_id[unit_is_in]

            intra_edges, inter_edges, global_global_edges, global_normal_edges, _ = edge_constructor(
                B_mini, batch_id_mini - bs_id_start, segment_ids_mini, X=X_mini, block_id=block_id_mini - offset)

            if not hasattr(edge_constructor, 'given_intra_edges'):
                mini_intra_edges.append(intra_edges + offset)
            if not hasattr(edge_constructor, 'given_inter_edges'):
                mini_inter_edges.append(inter_edges + offset)
            if global_global_edges is not None:
                mini_global_global_edges.append(global_global_edges + offset)
            if global_normal_edges is not None:
                mini_global_normal_edges.append(global_normal_edges + offset)
            offset += B_mini.shape[0]

        if hasattr(edge_constructor, 'given_intra_edges'):
            intra_edges = edge_constructor.given_intra_edges
        else:
            intra_edges = torch.cat(mini_intra_edges, dim=1)
        if hasattr(edge_constructor, 'given_inter_edges'):
            inter_edges = edge_constructor.given_inter_edges
        else:
            inter_edges = torch.cat(mini_inter_edges, dim=1)
        if global_global_edges is not None:
            global_global_edges = torch.cat(mini_global_global_edges, dim=1)
        if global_normal_edges is not None:
            global_normal_edges = torch.cat(mini_global_normal_edges, dim=1)

    return intra_edges, inter_edges, global_global_edges, global_normal_edges


class DenoisePretrainModel(nn.Module):

    def __init__(self, model_type, hidden_size, n_channel,
                 n_rbf=1, cutoff=7.0, n_head=1,
                 radial_size=16, edge_size=64, k_neighbors=9, n_layers=3,
                 n_residue_layers=2,
                 sigma_begin=10, sigma_end=0.01, n_noise_level=50,
                 dropout=0.1, std=10, global_message_passing=True,
                 atom_level=False, hierarchical=False, no_block_embedding=False,
                 gate_type='scalar', fps_ratio=0.5, pool_k=9) -> None:
        super().__init__()

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_channel = n_channel
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.n_head = n_head
        self.radial_size = radial_size
        self.edge_size = edge_size
        self.k_neighbors = k_neighbors
        self.n_layers = n_layers
        self.dropout = dropout
        self.std = std
        self.global_message_passing = global_message_passing
        self.atom_level = atom_level
        self.hierarchical = hierarchical
        self.no_block_embedding = no_block_embedding
        self.n_residue_layers = n_residue_layers
        self.residue_pool = False  # set True for models that pool to residue level
        self.graph_level_pred = False  # set True for models that return a fused graph_repr
        self.pair_type_encoder = False  # set True for models that need segment_ids per edge
        self.interface_biased_encoder = False  # set True for models that need segment_ids for biased sampling

        assert not (self.hierarchical and self.atom_level), 'Hierarchical model is incompatible with atom-level model'

        self.global_block_id = VOCAB.symbol_to_idx(VOCAB.GLB)

        self.block_embedding = BlockEmbedding(
            num_block_type=len(VOCAB),
            num_atom_type=VOCAB.get_num_atom_type(),
            num_atom_position=VOCAB.get_num_atom_pos(),
            embed_size=hidden_size,
            no_block_embedding=no_block_embedding
        )

        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=k_neighbors,
            global_message_passing=global_message_passing,
            global_node_id_vocab=[self.global_block_id],
            delete_self_loop=False)
        self.edge_embedding = nn.Embedding(4, edge_size)  # [intra / inter / global_global / global_normal]
        
        z_requires_grad = False
        if model_type == 'GET':
            from .GET.encoder import GETEncoder
            self.encoder = GETEncoder(
                hidden_size, radial_size, n_channel,
                n_rbf, cutoff, edge_size, n_layers,
                n_head, dropout=dropout,
                z_requires_grad=z_requires_grad
            )
        elif model_type == 'GETPool':
            from .GET.pool_encoder import GETPoolEncoder
            self.encoder = GETPoolEncoder(
                hidden_size, radial_size, n_channel,
                n_rbf, cutoff, edge_size, n_layers,
                n_head, dropout=dropout,
                z_requires_grad=z_requires_grad
            )
        elif model_type == 'SchNet':
            from .SchNet.encoder import SchNetEncoder
            self.encoder = SchNetEncoder(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetGeoAttn':
            from .SchNet.schnet_geo_attn import SchNetGeoAttnEncoder
            self.encoder = SchNetGeoAttnEncoder(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetGeoAttnV2':
            from .SchNet.schnet_geo_attn_v2 import SchNetGeoAttnV2Encoder
            self.encoder = SchNetGeoAttnV2Encoder(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetGeoAttnV3':
            from .SchNet.schnet_geo_attn_v3 import SchNetGeoAttnV3Encoder
            self.encoder = SchNetGeoAttnV3Encoder(hidden_size, edge_size, n_layers)
            self.pair_type_encoder = True
        elif model_type == 'SchNetGeoAttnV4':
            from .SchNet.schnet_geo_attn_v4 import SchNetGeoAttnV4Encoder
            self.encoder = SchNetGeoAttnV4Encoder(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetGeoAttnV5':
            from .SchNet.schnet_geo_attn_v5 import SchNetGeoAttnV5Encoder
            self.encoder = SchNetGeoAttnV5Encoder(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV10':
            from .SchNet.unet_encoder_v10 import SchNetUNetEncoderV10
            self.encoder = SchNetUNetEncoderV10(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV11':
            from .SchNet.unet_encoder_v11 import SchNetUNetEncoderV11
            self.encoder = SchNetUNetEncoderV11(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
        elif model_type == 'SchNetUNetV12':
            from .SchNet.unet_encoder_v12 import SchNetUNetEncoderV12
            self.encoder = SchNetUNetEncoderV12(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
        elif model_type == 'SchNetUNetV13':
            from .SchNet.unet_encoder_v13 import SchNetUNetEncoderV13
            self.encoder = SchNetUNetEncoderV13(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
        elif model_type == 'SchNetUNetV14':
            from .SchNet.unet_encoder_v14 import SchNetUNetEncoderV14
            self.encoder = SchNetUNetEncoderV14(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
        elif model_type == 'SchNetUNetV15':
            from .SchNet.unet_encoder_v15 import SchNetUNetEncoderV15
            self.encoder = SchNetUNetEncoderV15(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV16':
            from .SchNet.unet_encoder_v16 import SchNetUNetEncoderV16
            self.encoder = SchNetUNetEncoderV16(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
            self.interface_biased_encoder = True
        elif model_type == 'SchNetUNetV17':
            from .SchNet.unet_encoder_v17 import SchNetUNetEncoderV17
            self.encoder = SchNetUNetEncoderV17(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
        elif model_type == 'SchNetUNetV18':
            from .SchNet.unet_encoder_v18 import SchNetUNetEncoderV18
            self.encoder = SchNetUNetEncoderV18(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
            self.interface_biased_encoder = True
        elif model_type == 'SchNetUNetV19':
            from .SchNet.unet_encoder_v19 import SchNetUNetEncoderV19
            self.encoder = SchNetUNetEncoderV19(hidden_size, edge_size, n_layers,
                                                fps_ratio=fps_ratio, pool_k=pool_k,
                                                gate_type=gate_type)
        elif model_type == 'SchNetUNetV20':
            from .SchNet.unet_encoder_v20 import SchNetUNetEncoderV20
            self.encoder = SchNetUNetEncoderV20(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
            self.interface_biased_encoder = True
        elif model_type == 'SchNetUNetV21':
            from .SchNet.unet_encoder_v21 import SchNetUNetEncoderV21
            self.encoder = SchNetUNetEncoderV21(hidden_size, edge_size, n_layers,
                                                fps_ratio=fps_ratio, pool_k=pool_k,
                                                gate_type=gate_type)
            self.interface_biased_encoder = True
        elif model_type == 'SchNetUNetV23':
            from .SchNet.unet_encoder_v23 import SchNetUNetEncoderV23
            self.encoder = SchNetUNetEncoderV23(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
        elif model_type == 'SchNetUNetV24':
            from .SchNet.unet_encoder_v24 import SchNetUNetEncoderV24
            self.encoder = SchNetUNetEncoderV24(hidden_size, edge_size, n_layers,
                                                gate_type=gate_type)
        elif model_type == 'SchNetUNetV22Semantic':
            from .SchNet.unet_encoder_v22_semantic import SchNetUNetEncoderV22Semantic
            self.encoder = SchNetUNetEncoderV22Semantic(
                hidden_size, edge_size, n_layers,
                k_neighbors=k_neighbors,
                gate_type=gate_type,
                n_block_types=len(VOCAB),
            )
            self.residue_pool = True
        elif model_type == 'SchNetUNet':
            from .SchNet.unet_encoder import SchNetUNetEncoder
            self.encoder = SchNetUNetEncoder(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV2':
            from .SchNet.unet_encoder_v2 import SchNetUNetEncoderV2
            self.encoder = SchNetUNetEncoderV2(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV3':
            from .SchNet.unet_encoder_v3 import SchNetUNetEncoderV3
            self.encoder = SchNetUNetEncoderV3(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV4':
            from .SchNet.unet_encoder_v4 import SchNetUNetEncoderV4
            self.encoder = SchNetUNetEncoderV4(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV5':
            from .SchNet.unet_encoder_v5 import SchNetUNetEncoderV5
            self.encoder = SchNetUNetEncoderV5(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV6':
            from .SchNet.unet_encoder_v6 import SchNetUNetEncoderV6
            self.encoder = SchNetUNetEncoderV6(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV7':
            from .SchNet.unet_encoder_v7 import SchNetUNetEncoderV7
            self.encoder = SchNetUNetEncoderV7(hidden_size, edge_size, n_layers)
        elif model_type == 'SchNetUNetV8':
            from .SchNet.unet_encoder_v8 import SchNetUNetEncoderV8
            self.encoder = SchNetUNetEncoderV8(
                hidden_size, edge_size,
                n_atom_layers=n_layers,
                n_residue_layers=n_residue_layers,
                k_neighbors=k_neighbors,
                n_block_types=len(VOCAB),
            )
            self.residue_pool = True
        elif model_type == 'SchNetUNetV9':
            from .SchNet.unet_encoder_v9 import SchNetUNetEncoderV9
            self.encoder = SchNetUNetEncoderV9(
                hidden_size, edge_size,
                n_atom_layers=n_layers,
                n_residue_layers=n_residue_layers,
                k_neighbors=k_neighbors,
                n_block_types=len(VOCAB),
            )
            self.residue_pool = True
            self.graph_level_pred = True
        elif model_type == 'EGNN':
            from .EGNN.encoder import EGNNEncoder
            self.encoder = EGNNEncoder(hidden_size, edge_size, n_layers)
        elif model_type == 'DimeNet':
            from .DimeNet.encoder import DimeNetEncoder
            self.encoder = DimeNetEncoder(hidden_size, n_layers)
        elif model_type == 'TorchMD':
            from .TorchMD.encoder import TorchMDEncoder
            self.encoder = TorchMDEncoder(hidden_size, edge_size, n_layers)
        elif model_type == 'Equiformer':
            from .equiformer.encoder import EquiformerEncoder
            self.encoder = EquiformerEncoder(hidden_size, edge_size, n_head, n_layers)
        elif model_type == 'GemNet':
            from .gemnet.encoder import GemNetEncoder
            self.encoder = GemNetEncoder(hidden_size, radial_size, edge_size, n_layers, k_neighbors)
        elif model_type == 'MACE':
            from .MACE.encoder import MACEEncoder
            self.encoder = MACEEncoder(hidden_size, n_rbf, cutoff, n_layers)
        elif model_type == 'LEFTNet':
            from .LEFTNet.encoder import LEFTNetEncoder
            self.encoder = LEFTNetEncoder(hidden_size, n_rbf, cutoff, n_layers)
        else:
            raise NotImplementedError(f'Model type {model_type} not implemented!')
        
        if self.hierarchical:
            self.top_encoder = deepcopy(self.encoder)
        
        self.energy_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        if getattr(self, 'graph_level_pred', False):
            self.graph_energy_ffn = nn.Linear(hidden_size, 1, bias=False)

        # self.noise_level_ffn = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, n_noise_level)
        # )

        # TODO: add zero noise level
        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), n_noise_level)), dtype=torch.float)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # [n_noise_level]

    @torch.no_grad()
    def choose_receptor(self, batch_size, device):
        segment_retain = (torch.randn((batch_size, ), device=device) > 0).long()  # [bs], 0 or 1
        return segment_retain

    @torch.no_grad()
    def normalize(self, Z, B, block_id, batch_id, segment_ids, receptor_segment):
        # centering
        center = Z[(B[block_id] == self.global_block_id) & (segment_ids[block_id] == receptor_segment[batch_id][block_id])]  # [bs]
        Z = Z - center[batch_id][block_id]
        # normalize
        Z = Z / self.std
        return Z

    @torch.no_grad()
    def perturb(self, Z, block_id, batch_id, batch_size, segment_ids, receptor_segment):

        noise_level = torch.randint(0, self.sigmas.shape[0], (batch_size,), device=Z.device)
        # noise_level = torch.ones((batch_size, ), device=Z.device, dtype=torch.long) * (self.sigmas.shape[0] - 1)
        used_sigmas = self.sigmas[noise_level][batch_id]  # [Nb]
        used_sigmas = used_sigmas[block_id]  # [Nu]

        # randomly select one side to perturb (segment type 0 or segment type 1)
        perturb_block_mask = segment_ids == receptor_segment[batch_id]  # [Nb]
        perturb_mask = perturb_block_mask[block_id]  # [Nu]

        used_sigmas[~perturb_mask] = 0  # only one side of the complex is perturbed

        noise = torch.randn_like(Z)  # [Nu, channel, 3]

        Z_perturbed = Z + noise * used_sigmas.unsqueeze(-1).unsqueeze(-1)

        return Z_perturbed, noise, noise_level, perturb_mask
    
    @torch.no_grad()
    def update_global_block(self, Z, B, block_id):
        is_global = B[block_id] == self.global_block_id  # [Nu]
        scatter_ids = torch.cumsum(is_global.long(), dim=0) - 1  # [Nu]
        not_global = ~is_global
        centers = scatter_mean(Z[not_global], scatter_ids[not_global], dim=0)  # [Nglobal, n_channel, 3], Nglobal = batch_size * 2
        Z = Z.clone()
        Z[is_global] = centers
        return Z, not_global

    def pred_noise_from_energy(self, energy, Z):
        grad_outputs = [torch.ones_like(energy)]
        dy = grad(
            [energy],
            [Z],
            grad_outputs=grad_outputs,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]
        pred_noise = (-dy).view(-1, self.n_channel, 3).contiguous() # the direction of the gradients is where the energy drops the fastest. Noise adopts the opposite direction
        return pred_noise

    def get_edges(self, B, batch_id, segment_ids, Z, block_id):
        intra_edges, inter_edges, global_global_edges, global_normal_edges = construct_edges(
                    self.edge_constructor, B, batch_id, segment_ids, Z, block_id, complexity=2000**2)
        if self.global_message_passing:
            edges = torch.cat([intra_edges, inter_edges, global_global_edges, global_normal_edges], dim=1)
            edge_attr = torch.cat([
                torch.zeros_like(intra_edges[0]),
                torch.ones_like(inter_edges[0]),
                torch.ones_like(global_global_edges[0]) * 2,
                torch.ones_like(global_normal_edges[0]) * 3])
        else:
            edges = torch.cat([intra_edges, inter_edges], dim=1)
            edge_attr = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        edge_attr = self.edge_embedding(edge_attr)

        return edges, edge_attr

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=True, return_loss=True) -> ReturnValue:

        # batch_id and block_id
        with torch.no_grad():

            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            # Save residue-level info before atom_level collapses block_id to identity.
            if getattr(self, 'residue_pool', False):
                orig_block_id = block_id.clone()       # [Nu]  atom -> residue/fragment
                batch_id_coarse = batch_id.clone()     # [Nb]  residue -> graph
                B_coarse = B.clone()                   # [Nb]  block type per residue

                # if not getattr(self, '_residue_pool_checked', False):
                #     n_graphs = lengths.shape[0]
                #     # segment_ids here is still per-block [Nb]; 0 = protein, 1 = ligand
                #     prot = scatter_sum((segment_ids == 0).long(), batch_id, dim=0, dim_size=n_graphs)
                #     lig  = scatter_sum((segment_ids != 0).long(), batch_id, dim=0, dim_size=n_graphs)
                #     total = prot + lig
                #     expected_Nc = batch_id_coarse.shape[0]
                #     assert total.sum().item() == expected_Nc, (
                #         f'[V8 Debug] Block count mismatch: '
                #         f'prot+lig sum={total.sum().item()} != Nc={expected_Nc}'
                #     )
                #     print('\n[V8 Residue Pool Debug] Per-graph coarse node breakdown:', file=sys.stderr)
                #     for g in range(min(n_graphs, 5)):
                #         print(f'  graph {g}: {prot[g].item()} protein residues + '
                #               f'{lig[g].item()} ligand fragments = {total[g].item()} coarse nodes',
                #               file=sys.stderr)
                #     if n_graphs > 5:
                #         print(f'  ... ({n_graphs} graphs total)', file=sys.stderr)
                #     print(f'[V8 Residue Pool Debug] Total coarse nodes in batch: {expected_Nc} ✓\n',
                #           file=sys.stderr)
                #     self._residue_pool_checked = True

            if self.atom_level:  # this is for ablation
                # transform blocks to single units
                batch_id = batch_id[block_id]  # [Nu]
                segment_ids = segment_ids[block_id]  # [Nu]
                B = B[block_id]  # [Nu]
                block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]
            elif self.hierarchical:
                # transform blocks to single units
                bottom_batch_id = batch_id[block_id]  # [Nu]
                bottom_B = B[block_id]  # [Nu]
                bottom_segment_ids = segment_ids[block_id]  # [Nu]
                bottom_block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]

            batch_size = lengths.shape[0]
            # select receptor
            receptor_segment = self.choose_receptor(batch_size, batch_id.device)
            # normalize
            Z = self.normalize(Z, B, block_id, batch_id, segment_ids, receptor_segment)
            # perturbation
            Z_perturbed, noise, noise_level, perturb_mask = self.perturb(Z, block_id, batch_id, batch_size, segment_ids, receptor_segment)
            Z_perturbed, not_global = self.update_global_block(Z_perturbed, B, block_id)

        Z_perturbed.requires_grad_(True)

        # embedding
        # if not getattr(self, '_raw_dims_printed', False):
        #     Nb = B.shape[0]
        #     N  = A.shape[0]
        #     print('\n========== GET Raw Feature Dimensions ==========', file=sys.stderr)
        #     print(f'[Unit level - raw]', file=sys.stderr)
        #     print(f'  A (atom type):        shape={tuple(A.shape)},  vocab_size={VOCAB.get_num_atom_type()},  unique_in_batch={A.unique().numel()}', file=sys.stderr)
        #     print(f'  atom_positions:       shape={tuple(atom_positions.shape)},  vocab_size={VOCAB.get_num_atom_pos()},  unique_in_batch={atom_positions.unique().numel()}', file=sys.stderr)
        #     print(f'  Z (coords):           shape={tuple(Z_perturbed.shape)}  [N, n_channel, 3]', file=sys.stderr)
        #     print(f'[Block level - raw]', file=sys.stderr)
        #     print(f'  B (residue type):     shape={tuple(B.shape)},  vocab_size={len(VOCAB)},  unique_in_batch={B.unique().numel()}', file=sys.stderr)
        #     print(f'  block_id:             shape={tuple(block_id.shape)}  (maps each atom -> its block)', file=sys.stderr)
        #     print(f'  avg atoms per block:  {N / max(Nb, 1):.1f}', file=sys.stderr)
        #     print(f'[Graph level - raw]', file=sys.stderr)
        #     print(f'  batch_id:             shape={tuple(batch_id.shape)},  n_graphs={batch_id.max().item()+1}', file=sys.stderr)
        #     print(f'=================================================\n', file=sys.stderr)
        #     self._raw_dims_printed = True

        if self.hierarchical:
            bottom_H_0 = self.block_embedding.atom_embedding(A) + self.block_embedding.position_embedding(atom_positions)
            top_H_0 = 0 if self.block_embedding.no_block_embedding else self.block_embedding.block_embedding(B)
        else:
            H_0 = self.block_embedding(B, A, atom_positions, block_id)

        # encoding
        if self.hierarchical:
            # bottom level message passing
            edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, Z_perturbed, bottom_block_id)
            unit_repr, _, _, pred_Z = self.encoder(bottom_H_0, Z_perturbed, bottom_block_id, bottom_batch_id, edges, edge_attr)

            # top level message passing
            top_Z = scatter_mean(Z_perturbed if pred_Z is None else pred_Z, block_id, dim=0)  # [Nb, n_channel, 3]
            top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
            edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id)
            top_H_0 = top_H_0 + scatter_mean(unit_repr, block_id, dim=0)
            _, block_repr, graph_repr, _ = self.top_encoder(top_H_0, top_Z, top_block_id, batch_id, edges, edge_attr)
            unit_repr = unit_repr + block_repr[block_id]
        else:
            edges, edge_attr = self.get_edges(B, batch_id, segment_ids, Z_perturbed, block_id)
            if getattr(self, 'residue_pool', False):
                unit_repr, block_repr, graph_repr, pred_Z = self.encoder(
                    H_0, Z_perturbed, block_id, batch_id, edges, edge_attr,
                    orig_block_id=orig_block_id,
                    batch_id_coarse=batch_id_coarse,
                    B_coarse=B_coarse,
                )
            elif getattr(self, 'pair_type_encoder', False):
                unit_repr, block_repr, graph_repr, pred_Z = self.encoder(
                    H_0, Z_perturbed, block_id, batch_id, edges, edge_attr,
                    segment_ids=segment_ids,
                )
            elif getattr(self, 'interface_biased_encoder', False):
                unit_repr, block_repr, graph_repr, pred_Z = self.encoder(
                    H_0, Z_perturbed, block_id, batch_id, edges, edge_attr,
                    segment_ids=segment_ids,
                )
            else:
                unit_repr, block_repr, graph_repr, pred_Z = self.encoder(H_0, Z_perturbed, block_id, batch_id, edges, edge_attr)

        # predict energy
        # must be sum instead of mean! mean will make the gradient (predicted noise) pretty small, and the score net will easily converge to 0
        if getattr(self, 'graph_level_pred', False):
            pred_energy = self.graph_energy_ffn(graph_repr).squeeze(-1)   # [batch_size]
        else:
            pred_energy = scatter_sum(self.energy_ffn(block_repr).squeeze(-1), batch_id)

        # predict noise level
        # pred_noise_level = self.noise_level_ffn(graph_repr)  # [batch_size, n_noise_level]

        if return_noise or return_loss:
            # predict noise
            pred_noise = self.pred_noise_from_energy(pred_energy, Z_perturbed)
        else:
            pred_noise = None

        if return_loss:
            # print(pred_noise[perturb_mask][:10])
            perturb_mask = torch.logical_and(perturb_mask, not_global)  # do not calculate denoising loss on global nodes
            # noise loss
            noise_loss = F.mse_loss(pred_noise[perturb_mask], noise[perturb_mask], reduction='none')  # [Nperturb, n_channel, 3]
            noise_loss = noise_loss.sum(dim=-1).sum(dim=-1)  # [Nperturb]
            noise_loss = scatter_sum(noise_loss, batch_id[block_id][perturb_mask])  # [batch_size]
            noise_loss = 0.5 * noise_loss.mean()  # [1]

            # # align loss
            # align_loss = F.mse_loss(Z_perturbed[perturb_mask] - pred_Z[perturb_mask], noise[perturb_mask], reduction='none')  # [Nperturb, n_channel, 3]
            # align_loss = align_loss.sum(dim=-1).sum(dim=-1)  # [Nperturb]
            # align_loss = scatter_sum(align_loss, batch_id[block_id][perturb_mask])  # [batch_size]
            # align_loss = align_loss.mean()
            align_loss = 0

            # # noise level loss
            # noise_level_loss = F.cross_entropy(pred_noise_level, noise_level)
            noise_level_loss = 0

            # # punishments for trivial solution
            # punish_loss = torch.abs(pred_noise[perturb_mask] ** 2 - 1).sum()
            # print(len(pred_noise[perturb_mask]) * 3, punish_loss)

            # total loss
            loss = noise_loss # + noise_level_loss # + align_loss

        else:
            noise_loss, align_loss, noise_level_loss, loss = None, None, None, None

        return ReturnValue(

            # denoising variables
            energy=pred_energy,
            noise=pred_noise,
            noise_level=0,
            # noise_level=torch.argmax(pred_noise_level, dim=-1),

            # representations
            unit_repr=unit_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            loss=loss,
            noise_loss=noise_loss,
            noise_level_loss=noise_level_loss,
            align_loss=align_loss
        )