# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import pyximport

from graphormer.EHR_dataset import TadpoleDataset, MIMICDataset
from graphormer.utils.mask_utils import create_treat_mask_mimic, create_vals_mask_mimic

pyximport.install(setup_args={'include_dirs': np.get_include()})
from graphormer import algos
import time

def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.int32)
    x = x + feature_offset
    return x

def preprocess_item_ehr(item, offset=512, edges=False, data_path=None, bin_split_idx=0, one_node=False):
    start_p = time.time()
    edge_attr, edge_index, x, node_id = item.edge_attr, item.edge_index, item.x, item.node_id
    orig_x = x.clone()
    print("num nodes: ", x.shape[0])
    print("num edges: ", len(edge_attr))
    N = x.size(0)
    bin_feat = x[:,:bin_split_idx]
    reg_feat = x[:, bin_split_idx:]

    bin_feat = convert_to_single_emb(bin_feat, offset)
    x = torch.cat([bin_feat, reg_feat], dim=1)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    if not one_node:
        adj[edge_index[0, :], edge_index[1, :]] = True

    if edges:
        # edge feature here
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.int)
        if not one_node:
            attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr, offset=3) + 1

    start = time.time()
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    shortest_path_result = shortest_path_result.astype(np.int32)
    print("floyd_warshall: ", time.time()-start)
    if edges:
        max_dist = np.amax(shortest_path_result)
        start = time.time()
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        print("gen_edge_input: ", time.time() - start)
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.node_id = node_id
    item.orig_x = orig_x
    item.adj = adj
    item.attn_bias = attn_bias
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    if edges:
        item.edge_input = torch.from_numpy(edge_input).long()
        item.attn_edge_type = attn_edge_type
    print('full preprocessing: ', time.time() - start_p)
    if data_path and not one_node:
        torch.save(item, data_path)
    return item


def preprocess_item_ehr_mimic(item, data_path=None, edge_vars='age'):
    start_p = time.time()
    edge_attr, edge_index, vals, demographics, treatments = item.edge_attr, item.edge_index, item.vals, item.demographics, item.treatments
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:,None]
    if len(edge_attr) == 0:
        edge_attr = torch.ones([len(edge_index[1]), 1]) # for random edges include dummy edge features of 1
    print("num nodes: ", len(vals))
    print("num edges: ", len(edge_attr))
    N = len(vals)

    # add masking column for each feature column in vals
    for i, val in enumerate(vals):
        val_new = torch.zeros(val.shape[0], val.shape[1]*2)
        for idx in range(val.shape[1]):
            val_new[:,idx*2] = val[:, idx]
        vals[i] = val_new

    dem_embs = []
    for d in demographics:
        dem_embs.append(torch.cat([d[:,0:1], convert_to_single_emb(d[:,1:], offset = 6)], dim=1))

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    if len(edge_index) > 0:
        adj[edge_index[0, :], edge_index[1, :]] = True

    if len(edge_index) > 0:
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.int)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr, offset=3 if edge_vars=='age' else 200).int() + 1
    else:
        attn_edge_type = torch.zeros([N, N, 1], dtype=torch.int)

    start = time.time()
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    shortest_path_result = shortest_path_result.astype(np.int32)
    print("floyd_warshall: ", time.time()-start)

    max_dist = np.amax(shortest_path_result)
    start = time.time()
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    print("gen_edge_input: ", time.time() - start)

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.adj = adj
    item.demographics = torch.stack(dem_embs)
    item.attn_bias = attn_bias
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.attn_edge_type = attn_edge_type
    item.vals = vals
    print('full preprocessing: ', time.time() - start_p)
    torch.save(item, data_path)


def preprocess_item_mimic_gcn(item):
    edge_index, vals, demographics = item.edge_index, item.vals, item.demographics
    N = len(vals)

    # add masking column for each feature column in vals
    for i, val in enumerate(vals):
        val_new = torch.zeros(val.shape[0], val.shape[1]*2)
        for idx in range(val.shape[1]):
            val_new[:,idx*2] = val[:, idx]
        vals[i] = val_new

    dem_embs = []
    for d in demographics:
        dem_embs.append(torch.cat([d[:,0:1], convert_to_single_emb(d[:,1:], offset = 6)], dim=1))

    # combine
    item.demographics = torch.stack(dem_embs)
    item.edge_index = edge_index
    item.vals = vals
    return item

def add_masking(item, mask_ratio = 0.1):

    # mask further features with masked value in input and create update mask only true for masked values, do not mask missing values
    # use 0.0 and not 95 for continuous masked features (as linear layer can not deal with such a different value)
    missing_mask_disc = (item.orig_x[:,1:6] == item.mask_value)
    missing_mask_cont = (item.orig_x[:, 7:] == item.mask_value)
    item.x[:, 7:][missing_mask_cont] = -1.0 #make recognizable
    mask = torch.rand_like(item.x)
    mask = (mask < torch.tensor([mask_ratio])).bool()
    if item.not_mask_column_indices:
        item.not_mask_column_indices = [0,6]  # only age and gender
        mask[:, item.not_mask_column_indices] = False

    item.x[:,1:6][mask[:,1:6]] = item.mask_value  # mask discrete features with mask values
    item.x[:,7:][mask[:,7:]] = -1.0  # make recognizable

    all_masked_disc = (item.x[:,1:6] == item.mask_value)
    all_masked_cont = (item.x[:, 7:] == -1.0)
    final_mask_disc = torch.logical_and(all_masked_disc, ~missing_mask_disc)  # only values that were not missing before already
    final_mask_cont = torch.logical_and(all_masked_cont, ~missing_mask_cont)  # only values that were not missing before already

    # set missing continous values to 0
    item.x[:,7:][all_masked_cont] = 0.0
    mask[:, 1:6] = final_mask_disc
    mask[:, 7:] = final_mask_cont
    item.update_mask = mask
    return item

def add_masking_mimic(item, mask_ratio=0.15, block_size=6):
    # currently we mask 10% of the columns with 6 hour blocks

    # randomly select 10% of columns in vals and treatments to have missing blocks
    binary_treatment_mask = create_treat_mask_mimic(item, mask_ratio=mask_ratio, block_size=block_size)
    item.treatments[binary_treatment_mask] = 0 # after convert_to_single_embedding treatment values are never 0

    # for measurements
    binary_vals_mask_big, binary_mask_mask_big, final_vals_mask = create_vals_mask_mimic(item, mask_ratio=mask_ratio, block_size=block_size)
    # set masked values to 0 and corresponding is_masked column to 1

    item.vals[binary_vals_mask_big] = 0 # continuous values, no exact 0 values exist
    item.vals[binary_mask_mask_big] = 1 # mark as masked
    # only values that were not missing before already
    item.update_mask = [final_vals_mask[None], torch.from_numpy(binary_treatment_mask)[None]]
    return item


class MyTadpoleDataset(TadpoleDataset):
    def process(self):
        super(MyTadpoleDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx]).clone()
            item.idx = idx
            if self.mask:
                item = add_masking(item, mask_ratio=self.mask_ratio)
            return item
        else:
            return self.index_select(idx)


class MyMIMICDataset(MIMICDataset):
    def process(self):
        super(MyMIMICDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx]).clone()
            item.idx = idx
            if self.task.startswith('pre') and item.predict == False:
                item = add_masking_mimic(item, mask_ratio=self.mask_ratio, block_size=self.block_size)
            return item
        else:
            return self.index_select(idx)