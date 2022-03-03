# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# code was adapted for our data

import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_y_unsqueeze(x, padlen):
    # pad id = -1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype) - 1
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_mask_unsqueeze(x, padlen):
    # pad id = 0 -> will not be used for train/val/test, same as normal False nodes
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_pad_mask_unsqueeze(x, padlen):
    # pad id = 0 -> will not be used for train/val/test, same as normal False nodes
    x1, x2 = x.size()
    if x1 < padlen:
        new_x = x.new_zeros([padlen, x2], dtype=x.dtype)
        new_x[:x1] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_feat_graph_unsqueeze(x, padlen1, padlen2, padlen3, pad_mode):
    if pad_mode == 'original' or pad_mode == 'emb':
        x = x + 1
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)

def convert_to_single_emb(x, offset=512):
    feature_num = x.size(2)
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.int32)
    x = x + feature_offset
    return x

def pad_treat_unsqueeze(x, padlen1, padlen2, padlen3,):
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3], dtype=x.dtype) + 2 # padded elements are 2, classes 0 and 1
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    x = convert_to_single_emb(x, offset=3) # after this 0 will be free for masking
    return x.unsqueeze(0)

def pad_y_graph_unsqueeze(x, padlen1, padlen2, padlen3):
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

class Batch():
    def __init__(self, idx, y, update_mask, train_mask, val_mask, test_mask, node_id, attn_bias=None, attn_edge_type=None, spatial_pos=None,
                 in_degree=None, out_degree=None, edge_input=None, x=None, vals=None, treatments=None, demographics=None, is_measured=None, edge_index=None,
                 dev_mask = None, train_dev_mask=None, attn_mask=None, padding_mask=None):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x = x
        self.attn_mask = attn_mask
        # mimic
        self.vals = vals
        self.treatments = treatments
        self.is_measured = is_measured
        self.demographics = demographics
        self.y, self.node_id = y, node_id
        self.attn_bias, self.attn_edge_type, self.spatial_pos = attn_bias, attn_edge_type, spatial_pos
        self.edge_input = edge_input
        self.update_mask = update_mask
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.dev_mask = dev_mask
        self.train_dev_mask = train_dev_mask
        self.edge_index = edge_index
        self.padding_mask = padding_mask

    def to(self, device):
        self.idx = self.idx.to(device)
        if self.update_mask is not None: #None for classification task
            if type(self.update_mask) == list:
                self.update_mask[0] = self.update_mask[0].to(device)
                self.update_mask[1] = self.update_mask[1].to(device)
            else:
                self.update_mask = self.update_mask.to(device)
        self.train_mask = self.train_mask.to(device) if self.train_mask is not None else None
        self.val_mask = self.val_mask.to(device) if self.val_mask is not None else None
        self.test_mask = self.test_mask.to(device) if self.test_mask is not None else None
        self.dev_mask = self.dev_mask.to(device) if self.dev_mask is not None else None
        self.train_dev_mask = self.train_dev_mask.to(device) if self.train_dev_mask is not None else None
        if self.x is not None:
            self.x = self.x.to(device)
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(device)
        if self.vals is not None:
            self.vals = self.vals.to(device)
            self.treatments = self.treatments.to(device)
            self.is_measured = self.is_measured.to(device)
            self.demographics = self.demographics.to(device)
        if type(self.y) == list:
            self.y[0] = self.y[0].to(device)
            self.y[1] = self.y[1].to(device)
        else:
            self.y = self.y.to(device)
        if self.attn_edge_type is not None and self.edge_input is not None:
            self.attn_edge_type = self.attn_edge_type.to(device)
            self.edge_input = self.edge_input.to(device)
        if self.attn_bias is not None:
            self.attn_bias, self.spatial_pos = self.attn_bias.to(
                device), self.spatial_pos.to(device)
            self.in_degree, self.out_degree = self.in_degree.to(
                device), self.out_degree.to(device)
        if self.edge_index is not None:
            self.edge_index = [elem.to(device) for elem in self.edge_index] #self.edge_index.to(device)
        if self.padding_mask is not None:
            self.padding_mask = self.padding_mask.to(device)
        return self

    def __len__(self):
        if type(self.y) == list:
            return self.y[0].size(0)
        else:
            return self.y.size(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, dataset='tadpole', gcn=False, pad_mode='original'):
    if dataset == 'mimic':
        if gcn:
            items = [item for item in items if item is not None]
            items = [(item.idx, item.vals, item.treatments, item.edge_index,
                      item.demographics, item.is_measured, item.node_id, item.y,
                      item.update_mask, item.train_mask, item.val_mask, item.test_mask) for item in items]
            idxs, valss, treatmentss, edge_indexs, demographicss, is_measureds, node_ids, \
            ys, update_masks, train_masks, val_masks, test_masks = zip(*items)
            dev_masks, train_dev_masks = [None], [None]
        else:
            items = [item for item in items if item is not None]
            items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree, item.out_degree, item.vals, item.treatments,
                      item.demographics, item.is_measured, item.node_id, item.edge_input[:, :, :multi_hop_max_dist, :], item.y,
                      item.update_mask, item.train_mask, item.val_mask, item.test_mask, item.padding_mask) for item in
                     items]  # , item.edge_index
            idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, valss, treatmentss, demographicss, is_measureds, node_ids, \
            edge_inputs, ys, update_masks, train_masks, val_masks, test_masks, padding_masks = zip(*items)
            dev_masks, train_dev_masks = [None], [None]
    else:
        items = [item for item in items if item is not None and item.x.size(0) <= max_node]
        items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
                  item.out_degree, item.x, torch.tensor(item.node_id), item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.update_mask,
                  item.train_mask, item.val_mask, item.test_mask) for item in items]
        idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, node_ids, edge_inputs, ys, update_masks, train_masks, val_masks, test_masks \
            = zip(*items)

    if not gcn:
        for idx, _ in enumerate(attn_biases):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
        max_dist = max(i.size(-2) for i in edge_inputs)
    max_node_num = max(i.size(0) for i in valss) if dataset=='mimic' else max(i.size(0) for i in xs)
    max_hour_num = max(i.size(1) for i in valss) if dataset == 'mimic' else None

    if dataset == 'mimic': #want to do batching
        val_feature_num = valss[0].shape[2]
        if type(ys[0]) == tuple: # for masking we have two ys for val and treatments
            vals_y, treat_y = zip(*ys)
            y = [torch.cat([pad_y_graph_unsqueeze(i, max_node_num, max_hour_num, vals_y[0].shape[2]) for i in vals_y]),
                 torch.cat([pad_y_graph_unsqueeze(i, max_node_num, max_hour_num, 16) for i in treat_y])]
        else:
            if len(ys[0].shape) == 1:
                y = torch.cat([pad_y_unsqueeze(i, max_node_num) for i in ys]) # padded ys are set to -1 just to make sure an error would occur if they are used (no binary classification anymore)
            else:
                y = torch.cat([pad_pad_mask_unsqueeze(i, max_node_num) for i in ys])
    else:
        y = torch.cat(ys)
    node_id = torch.cat([pad_y_unsqueeze(i, max_node_num) for i in node_ids])

    if dataset == 'mimic':
        vals = torch.cat([pad_feat_graph_unsqueeze(i, max_node_num, max_hour_num, val_feature_num, pad_mode) for i in valss]) #max_num_hours, num_features
        is_measured = torch.cat([pad_feat_graph_unsqueeze(i, max_node_num, max_hour_num, 56, pad_mode) for i in is_measureds])

        demographics = torch.cat([pad_2d_unsqueeze(i.squeeze(1), max_node_num) for i in demographicss]) # could also delete +1 but do not need to

        if pad_mode == 'pad_emb' or pad_mode == 'emb':
            treatments = torch.cat([pad_treat_unsqueeze(i, max_node_num, max_hour_num, 16) for i in treatmentss])
        else:
            treatments = torch.cat([pad_feat_graph_unsqueeze(i, max_node_num, max_hour_num, 16, pad_mode) for i in treatmentss])

    else:
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    if update_masks[0] is not None: #None for classification task
        if type(update_masks[0]) == list: # in mimic we have two update masks for val and treatments
            vals_update_masks, treat_update_masks = zip(*update_masks)
            update_mask = [torch.cat([pad_y_graph_unsqueeze(i.squeeze(), max_node_num, max_hour_num, 56) for i in vals_update_masks]),
                           torch.cat([pad_y_graph_unsqueeze(i.squeeze(), max_node_num, max_hour_num, 16) for i in treat_update_masks])]
        else:
            update_mask = torch.cat(update_masks)
    else:
        update_mask = None
    train_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in train_masks]) if train_masks[0] is not None else None
    val_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in val_masks]) if val_masks[0] is not None else None
    test_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in test_masks]) if test_masks[0] is not None else None

    if dataset == 'mimic':
        dev_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in dev_masks]) if dev_masks[0] is not None else None
        train_dev_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in train_dev_masks]) if train_dev_masks[0] is not None else None
        if not gcn:
            padding_mask = torch.cat([pad_pad_mask_unsqueeze(i, max_node_num) for i in padding_masks]) if padding_masks[0] is not None else None
    if not gcn:
        # these are Graphormer related inputs, they are all processed by an embedding layer so they can work with the regular +1 padding
        edge_input = torch.cat([pad_3d_unsqueeze(
            i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
        attn_bias = torch.cat([pad_attn_bias_unsqueeze(
            i, max_node_num + 1) for i in attn_biases])
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
        spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                            for i in spatial_poses])
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                              for i in in_degrees])
        out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                               for i in out_degrees])

    if dataset == 'mimic':
        if gcn:
            return Batch(
                idx=torch.LongTensor(idxs),
                vals=vals,
                treatments=treatments,
                demographics=demographics,
                is_measured=is_measured,
                edge_index=edge_indexs,
                y=y,
                update_mask=update_mask,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                node_id=node_id
            )
        else:
            return Batch(
                idx=torch.LongTensor(idxs),
                attn_bias=attn_bias,
                attn_edge_type=attn_edge_type,
                spatial_pos=spatial_pos,
                in_degree=in_degree,
                out_degree=out_degree,
                vals=vals,
                treatments = treatments,
                demographics = demographics,
                is_measured = is_measured,
                edge_input=edge_input,
                y=y,
                update_mask=update_mask,
                train_mask = train_mask,
                val_mask = val_mask,
                test_mask = test_mask,
                dev_mask = dev_mask,
                train_dev_mask = train_dev_mask,
                node_id = node_id,
                padding_mask = padding_mask
            )
    else:
        return Batch(
            idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=out_degree,
            x=x,
            edge_input=edge_input,
            y=y,
            update_mask=update_mask,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            node_id=node_id
        )


def collator_no_edge_feat(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.y) for item in items]
    idxs, attn_biases, spatial_poses, in_degrees, out_degrees, xs, ys = zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in out_degrees])
    return Batch(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=None,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=None,
        y=y,
    )
