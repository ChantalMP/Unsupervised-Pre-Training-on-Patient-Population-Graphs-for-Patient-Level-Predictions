import json
import os
import os.path as osp
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data, RandomNodeSampler

from graphormer import wrapper


class EHRDataset(Dataset):
    def __init__(self, root, mask, name, raw_file_name, offset=3, bin_split_idx=14, transform=None, pre_transform=None, parts=None, split=None,
                 drop_val_patients=False, cross_val_split=None, fold=None, sim_graph=True, mask_all=False, k=5, mask_ratio=0.1):
        # if true masking task, else classification
        self.mask = mask
        self.mask_value = 95  # no class label but inside offset
        self.parts = parts
        part_str = f'_truncated_{parts}parts' if parts else ''
        self.root = root
        self.raw_file_name = raw_file_name
        self.full_data = (raw_file_name == "tadpole_numerical_full.csv")
        self.tranductive_pre = False
        self.k = k
        self.mask_ratio = mask_ratio
        self.cross_val = False
        self.sim_graph = True
        self.mask_all = True

        self.discrete_columns =  ['node_ID', 'DX_bl','PTGENDER', 'APOE4', 'CDRSB','ADAS11', 'MMSE','RAVLT_immediate']
        self.data_path = f'{name}_graph_class{part_str}{f"_drop_val_{split}" if drop_val_patients else ""}_fold{fold}{"_full" if self.full_data else ""}{"_transductive_pre" if self.tranductive_pre else ""}{"_sim" if self.sim_graph else ""}{f"_k{self.k}" if self.k != 5 else ""}.pt'
        self.drop_val_patients = drop_val_patients



        if split:
            self.split = split
            self.test_idxs = np.load('data/tadpole/split/test_idxs.npy')
            if cross_val_split is None:
                self.train_idxs = np.load(f'data/tadpole/split/cross_val/train_idxs_fold{fold}_strat.npy')
                self.val_idxs = np.load(f'data/tadpole/split/cross_val/val_idxs_fold{fold}_strat.npy')
            else:
                self.cross_val = True
                self.cross_val_split = cross_val_split
                self.train_idxs = np.load(f'data/tadpole/split/cross_val/train_idxs_fold{cross_val_split}_strat.npy')
                self.val_idxs = np.load(f'data/tadpole/split/cross_val/val_idxs_fold{cross_val_split}_strat.npy')
        else:
            self.split = None

        super().__init__(root, transform, pre_transform)

        processed_data = torch.load(osp.join(self.processed_dir, self.data_path))
        pre_processed_data_path = Path(osp.join(self.processed_dir, f'{name}_graph_class_preprocessed{part_str}'
                                                                    f'{f"_drop_val_{split}" if drop_val_patients else ""}_fold{fold}{"_full" if self.full_data else ""}'
                                                                    f'{"_transductive_pre" if self.tranductive_pre else ""}{"_sim" if self.sim_graph else ""}{f"_k{self.k}" if self.k != 5 else ""}.pt'))

        if pre_processed_data_path.exists():
            self.pre_processed_data = torch.load(pre_processed_data_path)
        else:
            self.pre_processed_data = wrapper.preprocess_item_ehr(item=processed_data, offset=offset, edges=True, data_path=pre_processed_data_path,
                                                                  bin_split_idx=bin_split_idx)

    @property
    def raw_file_names(self):
        return [self.raw_file_name]

    @property
    def processed_file_names(self):
        return [self.data_path]

    def create_edges(self, df, age_key, sex_key, feature_type=1):
        # edge_index
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edges = []
        edge_features = []
        for idx1, node1 in enumerate(df.iterrows()):
            if idx1 % 100 == 0:
                print(idx1)
            for idx2, node2 in enumerate(df.iterrows()):
                if idx1 != idx2:
                    age_diff = round(abs(node1[1][age_key] - node2[1][age_key]))
                    sex_diff = abs(node1[1][sex_key] - node2[1][sex_key])
                    if age_diff <= 2:
                        if feature_type == 1:  # 1 feature for age and sex together
                            weight = 3 - age_diff  # 3 for same age, 2 for 1 year diff, 1 for 2 years diff and diff gender
                            if sex_diff == 0:
                                weight += 3  # 6 for same age, 5 for 1 year diff, 4 for 2 years diff and same gender
                            edge_features.append(int(weight))
                        elif feature_type == 2:  # seperate features for age and sex
                            age_weight = 3 - age_diff
                            sex_weight = 1 - sex_diff  # one if same gender, else 0
                            weight = torch.tensor([age_weight, sex_weight])
                            edge_features.append(weight.int())

                        # add edge if age difference small enough
                        edges.append((idx1, idx2))

        edge_idx = torch.tensor(edges, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.int32) if feature_type == 1 else torch.stack(edge_features)
        return edge_idx, edge_features

    def create_x_y(self, X_norm, df, label_key):

        # x: node feature matrix [num_nodes, num_node_features]
        x = torch.tensor(X_norm.values.astype(np.float32))

        # replace all missing values with mask value
        missing_mask = torch.isnan(x)
        x[missing_mask] = self.mask_value

        # y: Target to train against, node-level targets of shape [num_nodes, *]
        if self.mask:
            y = x.clone()  # input features should be predicted, no features masked yet
        else:
            y = torch.tensor(df[label_key].values, dtype=torch.float)

        return x, y

    def save_data(self, node_id, x, edge_idx, edge_features, y, final_mask, train_mask, val_mask, test_mask, not_mask_column_indices, node_idx=None,
                  split=None):
        data = Data(node_id=node_id, x=x, edge_index=edge_idx.t().contiguous(), edge_attr=edge_features, y=y, update_mask=final_mask,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, mask_value=self.mask_value,
                    not_mask_column_indices=not_mask_column_indices, split=split)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        if node_idx is not None:
            torch.save(data, osp.join(self.processed_dir, f'one_node_graphs{"" if self.mask else "_class"}/graph_{node_idx}'))
        else:
            torch.save(data, osp.join(self.processed_dir, self.data_path))

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = self.pre_processed_data
        if self.mask:
            data.y = data.orig_x.clone()
        return data


class TadpoleDataset(EHRDataset):

    def tadpole_train_val_test_masks(self, df):
        train_idxs, val_idxs, test_idxs = self.train_idxs, self.val_idxs, self.test_idxs
        if self.drop_val_patients:
            train_mask = torch.zeros(len(self.train_idxs), dtype=torch.bool)
            val_mask = torch.zeros(len(self.train_idxs) + len(self.val_idxs), dtype=torch.bool)
            test_mask = torch.zeros(len(self.train_idxs) + len(self.test_idxs), dtype=torch.bool)
            train_idxs = [df.index.get_loc(df.index[df.node_ID == train_idx][0]) for train_idx in train_idxs] if self.split == 'train' else []
            val_idxs = [df.index.get_loc(df.index[df.node_ID == val_idx][0]) for val_idx in val_idxs] if self.split == 'val' else []
            test_idxs = [df.index.get_loc(df.index[df.node_ID == test_idx][0]) for test_idx in test_idxs] if self.split == 'test' else []

        else:
            train_mask = torch.zeros(564, dtype=torch.bool) if not self.tranductive_pre else torch.ones(564,
                                                                                                        dtype=torch.bool)  # pretrain with all nodes
            val_mask = torch.zeros(564, dtype=torch.bool) if not self.tranductive_pre else torch.ones(564, dtype=torch.bool)  # no validation set
            test_mask = torch.zeros(564, dtype=torch.bool)
        train_mask[train_idxs] = True
        val_mask[val_idxs] = True
        test_mask[test_idxs] = True

        return train_mask, val_mask, test_mask

    def compute_dem_similarity(self, node1_age, node2_age, node1_sex, node2_sex, node1_apoe, node2_apoe):
        age_diff = round(abs(node1_age - node2_age))
        sex_diff = abs(node1_sex - node2_sex)
        apoe_diff = abs(node1_apoe == node2_apoe)
        return torch.tensor(((age_diff <= 2)+(1-sex_diff)+apoe_diff)/3)

    def compute_cog_test_similarity(self, node1_cog, node2_cog):
        if np.isnan(node1_cog).any() or np.isnan(node2_cog).any():
            nan_mask = ~np.logical_or(np.isnan(node1_cog), np.isnan(node2_cog))
            dist = torch.tensor(np.linalg.norm(node1_cog[nan_mask] - node2_cog[nan_mask], axis=0))
        else:
            dist = torch.tensor(np.linalg.norm(node1_cog - node2_cog, axis=0))
        return 1 -(dist-0)/(116-0) # normalized

    def compute_imaging_similarity(self, node1_img, node2_img):
        dist = torch.tensor(np.linalg.norm(node1_img - node2_img, axis=0))
        return 1 - torch.sigmoid(dist)

    def create_edges_similarity_tadpole(self, df):
        # edge_index
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edges = []
        edge_features = []
        ages = {}
        sexs = {}
        apoes = {}
        cogs = {}
        imgs = {}
        for idx1, node1 in enumerate(df.iterrows()):
            if idx1 in ages:
                node1_age = ages[idx1]
                node1_sex = sexs[idx1]
                node1_apoe = apoes[idx1]
                node1_cog = cogs[idx1]
                node1_img = imgs[idx1]
            else:
                ages[idx1] = node1[1]['AGE']
                sexs[idx1] = node1[1]['PTGENDER']
                apoes[idx1] = node1[1]['APOE4']
                cogs[idx1] = node1[1][['CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']].values
                imgs[idx1] = node1[1][['Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG']].values
                node1_age = ages[idx1]
                node1_sex = sexs[idx1]
                node1_apoe = apoes[idx1]
                node1_cog = cogs[idx1]
                node1_img = imgs[idx1]

            all_sims = {}
            if idx1 % 100 == 0:
                print(idx1)
            for idx2, node2 in enumerate(df.iterrows()):
                if idx1 != idx2:

                    if idx2 in ages:
                        node2_age = ages[idx2]
                        node2_sex = sexs[idx2]
                        node2_apoe = apoes[idx2]
                        node2_cog = cogs[idx2]
                        node2_img = imgs[idx2]
                    else:
                        ages[idx2] = node2[1]['AGE']
                        sexs[idx2] = node2[1]['PTGENDER']
                        apoes[idx2] = node2[1]['APOE4']
                        cogs[idx2] = node2[1][['CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']].values
                        imgs[idx2] = node2[1][['Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG']].values
                        node2_age = ages[idx2]
                        node2_sex = sexs[idx2]
                        node2_apoe = apoes[idx2]
                        node2_cog = cogs[idx2]
                        node2_img = imgs[idx2]

                    # compute all similarity features
                    dem_similarity = self.compute_dem_similarity(node1_age, node2_age, node1_sex, node2_sex, node1_apoe, node2_apoe)
                    cog_similarity = self.compute_cog_test_similarity(node1_cog, node2_cog)
                    imaging_similarity = self.compute_imaging_similarity(node1_img, node2_img)

                    all_sims[idx2] = [np.mean([dem_similarity, cog_similarity, imaging_similarity]), dem_similarity, cog_similarity, imaging_similarity]

            sorted_sims = sorted(all_sims.items(), key=lambda x: x[1][0], reverse=True)
            for i in range(self.k): # add 5 neighrest neighbours to edges
                edges.append((idx1, sorted_sims[i][0]))
                edge_features.append(
                    torch.tensor([torch.round(sorted_sims[i][1][1]*3), torch.round(sorted_sims[i][1][2]*100), torch.round(sorted_sims[i][1][3]*100)],
                                 dtype=torch.int))

        edge_idx = torch.tensor(edges, dtype=torch.long)
        edge_features = torch.stack(edge_features)
        return edge_idx, edge_features

    def process(self):
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            df = pd.read_csv(raw_path)

            # only select labels of start set
            if not self.full_data:
                df = df[['node_ID', 'DX_bl', 'AGE', 'PTGENDER', 'APOE4', 'CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain',
                         'Entorhinal','MidTemp', 'FDG']]

            if self.drop_val_patients:
                if self.split == 'train':
                    idxs = self.train_idxs
                elif self.split == 'val':
                    idxs = np.concatenate((self.val_idxs, self.train_idxs))
                else:  # test split
                    idxs = np.concatenate((self.test_idxs, self.train_idxs))

                # inductive: normalize only given training data
                for col in [col for col in df.columns if col not in self.discrete_columns]:
                    df[col] = (df[col] - df[col][self.train_idxs].min()) / (df[col][self.train_idxs].max() - df[col][self.train_idxs].min())

                df = df[df.node_ID.isin(idxs)]

            else:
                # transductive: normalize on all nodes
                for col in [col for col in df.columns if col not in self.discrete_columns]:
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

            node_ids = np.array(df['node_ID'].values)
            # drop labels and ids
            drop_columns = ['DX_bl', 'node_ID']

            X = df.drop(drop_columns, axis=1)

            # discrete_columns without dropped ones
            discrete_features = [col for col in self.discrete_columns if col not in drop_columns]
            # sort columns so discrete features are in the front
            X = X[discrete_features + [col for col in X.columns if col not in discrete_features]]

            if not self.mask_all:
                mask_columns = ['APOE4', 'CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']
                not_mask_column_indices = np.where(np.isin(X.columns, mask_columns, invert=True))
            else:
                mask_columns = ['APOE4', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain', 'Entorhinal','MidTemp', 'FDG']
                not_mask_column_indices = np.where(np.isin(X.columns, mask_columns, invert=True)) #dont mask age and gender

            x, y = self.create_x_y(X_norm=X, df=df, label_key='DX_bl')

            train_mask, val_mask, test_mask = self.tadpole_train_val_test_masks(df)

            edge_idx, edge_features = self.create_edges_similarity_tadpole(df)

            self.save_data(node_id=node_ids, x=x, edge_idx=edge_idx, edge_features=edge_features, y=y, final_mask=None, train_mask=train_mask,
                           val_mask=val_mask, test_mask=test_mask, not_mask_column_indices=not_mask_column_indices, split=self.split)


class MIMICDataset(Dataset):
    def __init__(self, root, drop_val_patients=False, use_treatment_input=True, task="", split='train', transform=None, pre_transform=None, edge_vars='age',
                 num_graphs=43, gcn=False, mlp=False, rotation=0, predict=False, pad_mode='original', mask_ratio=0.15, block_size=6, k=5):
        # if true masking task, else classification
        self.mask_value = 0
        self.my_root = root
        self.split = split
        self.rotation = rotation
        self.gcn = gcn
        self.mlp = mlp
        self.pad_mode = pad_mode
        self.predict = predict
        self.drop_val_patients = drop_val_patients
        self.vals_weight = np.load('data_vis/MI_scores/score_array_vals.npy')
        self.treat_weight = np.load('data_vis/MI_scores/score_array_treat.npy')
        self.dems_weight = np.load('data_vis/MI_scores/score_array_dems.npy')[[0,1,4,5]]
        self.edge_vars = edge_vars # 0:'age', 1:'half_edge_half_node', 2:'half_edge_full_node', 3:'full_edge_full_node', 4: 'dems', 5:only vals
        edge_id = 3 if self.edge_vars == 'full_edge_full_node' else (2 if self.edge_vars == 'half_edge_full_node' else (1 if self.edge_vars == 'half_edge_half_node' else (4 if self.edge_vars == 'dems' else 0)))
        self.k = k
        if self.edge_vars == 'none':
            edge_id = 'none'
        elif self.edge_vars == 'vals':
            edge_id = f'5_k{self.k}' if self.k != 5 else 5

        self.lin_interpolation = True
        self.mask_col = (self.pad_mode == 'pad_emb')
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        if self.split == 'test':
            self.all_graph_idx_files = [f'rot{rotation}/random_graph_subset_{i}.json' for i in range(num_graphs)]
            self.all_graphs = [f'rotations/rot{rotation}/mimic_graph_test_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt' for i in range(num_graphs)]
            self.all_pre_processed_graphs = [
                f'rotations/rot{rotation}/mimic_graph_test_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt' for i in range(num_graphs)]
            self.data_path = f'rotations/rot{rotation}/mimic_graph_test_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_'

        else:
            self.all_graph_idx_files = [f'rot{rotation}/random_graph_subset_dev_{i}.json' for i in
                                        range(num_graphs)]  # 55 for age graph, 43 for random graph
            self.all_graphs = [
                f'rotations/rot{rotation}/mimic_graph_full_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt' if not drop_val_patients or self.split != 'train' else f'rotations/rot{rotation}/mimic_graph_train_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt' for i in
                range(num_graphs)]
            self.all_pre_processed_graphs = [
                f'rotations/rot{rotation}/mimic_graph_full_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt' if not drop_val_patients or self.split != 'train'
                else f'rotations/rot{rotation}/mimic_graph_train_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt'
                for i in range(num_graphs)]
            self.data_path = f'rotations/rot{rotation}/mimic_graph_full_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_' if not self.drop_val_patients or self.split != 'train' else f'rotations/rot{rotation}/mimic_graph_train_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_'
        self.use_treatment_input = use_treatment_input
        self.task = task  # possible tasks: mask_random, mask_next_step, mask_treatment, icd, los, rea, acu
        self.stay_lengths = {'los': 24}
        self.mask = task in ['mask_random', 'mask_next_step', 'mask_treatment']

        super().__init__(root + f'/{split}', transform, pre_transform)

        if not gcn:
            for graph in self.all_graphs:
                subset = graph.split('.')[0].split('_')[-1]
                processed_data = torch.load(osp.join(self.processed_dir, graph))
                if self.split == 'test':
                    pre_processed_data_path = Path(osp.join(self.processed_dir, f'rotations/rot{rotation}/mimic_graph_test_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{subset}.pt'))

                else:
                    pre_processed_data_path = Path(osp.join(self.processed_dir,
                                                            f'rotations/rot{rotation}/mimic_graph_full_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{subset}.pt' if not self.drop_val_patients or self.split != 'train'
                                                            else f'rotations/rot{rotation}/mimic_graph_train_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{subset}.pt'))

                if not pre_processed_data_path.exists():
                    # preprocess and save results
                    wrapper.preprocess_item_ehr_mimic(item=processed_data, data_path=pre_processed_data_path, edge_vars=self.edge_vars)

    @property
    def raw_file_names(self):
        return self.all_graph_idx_files

    @property
    def processed_file_names(self):
        return self.all_graphs

    def train_val_test_masks(self, splits):
        train_mask, val_mask, test_mask, train_dev_mask, dev_mask = None, None, None, None, None
        if self.drop_val_patients:  # inductive
            if self.split == 'train':
                # get all elements in split where second element is 'train'
                train_data = [split for split in splits if split[1] == 'train' or split[1] == 'dev']
                train_mask = torch.tensor([split[1] == 'train' or split[1] == 'dev' for split in train_data]) #for regular training ignore dev set
                train_dev_mask = torch.tensor([split[1] == 'train' for split in splits])

            elif self.split == 'val':
                # get all elements in split where second element is 'val' or 'train'
                val_data = [split for split in splits if split[1] == 'val' or split[1] == 'train' or split[1] == 'dev']
                train_mask = torch.tensor([split[1] == 'train' or split[1] == 'dev' for split in val_data])
                train_dev_mask = torch.tensor([split[1] == 'train' for split in splits])
                dev_mask = torch.tensor([split[1] == 'dev' for split in val_data])
                val_mask = torch.tensor([split[1] == 'val' for split in val_data])

            elif self.split == 'test':
                # get all elements in split where second element is 'test' or 'train' or 'val'
                test_data = [split for split in splits if split[1] == 'test' or split[1] == 'train' or split[1] == 'val' or split[1] == 'dev']
                test_mask = torch.tensor([split[1] == 'test' for split in test_data])

        return train_mask, val_mask, test_mask, train_dev_mask, dev_mask

    def patient_part_of_split(self, patient_split):
        if self.drop_val_patients:
            if patient_split == 'train' or patient_split == 'dev':
                return self.split == 'train' or self.split == 'val' or self.split == 'test'
            elif patient_split == 'val':
                return self.split == 'val' or self.split == 'test'
            elif patient_split == 'test':
                return self.split == 'test'
        else:
            return self.split == 'train' or self.split == 'val' or self.split == 'test'  # for now we do not use test set nodes at all

    def get_or_compute_val_descriptor(self, idx, vals, descriptor_cache):
        if idx in descriptor_cache:
            return descriptor_cache[idx]
        else:
            descriptors = np.stack([vals.mean(axis=0),vals.std(axis=0),vals.min(axis=0)[0],vals.max(axis=0)[0]])
            descriptor_cache[idx] = descriptors
            return descriptors

    def compute_vals_similarity(self, vals_descriptors1, vals_descriptors2):
        # per column compute similarity of time-series
        dist = torch.tensor(np.linalg.norm(vals_descriptors1-vals_descriptors2, axis=0).mean())
        return 1- torch.sigmoid(dist) # in which range is this value? in test graph: 0.01 - 0.42 for half features, 0.02 - 0.4 for full features

    def create_edges_feat_similarity_knn(self, patient_half_vals):
        # edge_index
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edges = []
        edge_features = []
        descriptor_cache = {}
        for idx1, half_vals in enumerate(patient_half_vals):
            all_sims = {}
            if idx1 % 100 == 0:
                print(idx1)
            for idx2, half_vals2 in enumerate(patient_half_vals):
                if idx1 != idx2:
                    vals_descriptors1 = self.get_or_compute_val_descriptor(idx1, half_vals, descriptor_cache)
                    vals_descriptors2 = self.get_or_compute_val_descriptor(idx2, half_vals2, descriptor_cache)
                    vals_similarity = self.compute_vals_similarity(vals_descriptors1, vals_descriptors2)
                    all_sims[idx2] = vals_similarity

            # compute 5 nearest neighbors and add edges
            sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)
            for i in range(self.k):
                edges.append((idx1, sorted_sims[i][0]))
                edge_features.append(torch.round((sorted_sims[i][1]*100)).int()) #discretize similarity to 0-17


        edge_idx = torch.tensor(edges, dtype=torch.long)
        edge_features = torch.stack(edge_features)[:,None]
        return edge_idx, edge_features


    def save_data(self, node_id, vals, demographics, treatments, is_measured, edge_idx, edge_features, y, final_mask, train_mask, val_mask, test_mask,
                  train_dev_mask, dev_mask, subset_id):
        data = Data(node_id=node_id, vals=vals, demographics=demographics, treatments=treatments, is_measured=is_measured,
                    edge_index=edge_idx.t().contiguous(), edge_attr=edge_features, y=y, update_mask=final_mask, train_mask=train_mask,
                    train_dev_mask=train_dev_mask, dev_mask=dev_mask, val_mask=val_mask, test_mask=test_mask, split=self.split, mask_value=self.mask_value)

        # create rotation dataset if it does not exist
        if not os.path.exists(osp.join(self.processed_dir, f'rot{self.rotation}')):
            os.makedirs(osp.join(self.processed_dir, f'rot{self.rotation}'))

        torch.save(data, osp.join(self.processed_dir, self.data_path + f'{subset_id}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.gcn or self.mlp:
            data = torch.load(osp.join(self.processed_dir, self.all_graphs[idx]))
            data = wrapper.preprocess_item_mimic_gcn(item=data)
            data.num_nodes = len(data.vals)
        else:
            data = torch.load(osp.join(self.processed_dir, self.all_pre_processed_graphs[idx]))
        window_length = 24
        # form y vector given task
        if self.task == 'pre_mask':
            y = (torch.stack([d['vals'][:window_length] for d in data.y]), torch.stack([d['treatments_hour'][:window_length] for d in data.y]))
        else:
            y = torch.tensor([d[self.task] for d in data.y])

                # for training, select random stay length later, for val task dependent, but for now just always use 24h, and only perform tasks on first 24h

        treatments = torch.stack([d[:window_length] for d in data['treatments']])
        is_measured = torch.stack([d[:window_length] for d in data['is_measured']])
        vals = torch.stack([d[:window_length] for d in data['vals']])
        data.y = y

        # set changed variables in data
        data.vals = vals
        data.treatments = treatments
        data.is_measured = is_measured
        data.predict = self.predict
        return data

    def process(self):
        for raw_path in self.raw_paths:
            subset_id = raw_path.split('.')[0].split('_')[-1]
            # Read data from `raw_path`.
            patient_icuids_and_splits = json.load(open(raw_path, 'r'))  # gives list of patients that should be in graph together with their split

            # Keep vals and treatments separate as we sometimes want treatments in training data and sometimes not
            patient_icuids = []
            patient_vals = []
            patient_half_vals = []
            patient_half_is_measured = []
            patient_half_vals_node = []
            patient_half_is_measured_node = []
            patient_dems = []
            patient_graph_dems = []
            patient_treatments = []
            patient_is_measured = []
            patient_ys = []  # dict for several prediction tasks
            # go through all patient folders with id in patient_icuids
            for info in patient_icuids_and_splits:
                if self.rotation == 0:
                    patient_icuid, split = info
                    folder = split
                else:
                    patient_icuid, split, folder = info

                patient_icuid = int(patient_icuid)
                if self.patient_part_of_split(split):
                    if folder == 'dev':
                        folder = 'train'
                    patient_dir = osp.join(self.my_root, folder, f'patient_{patient_icuid}') # for other rotations than 0 the patient might not be saved in the splits folder, because all is saved according to rotation 0
                    statics = pd.read_csv(patient_dir + '/' + 'statics.csv')
                    ts_vals = pd.read_csv(patient_dir + '/' + 'ts_vals_linear_imputed.csv') if self.lin_interpolation else pd.read_csv(patient_dir + '/' + 'ts_vals.csv')
                    ts_treatment = pd.read_csv(patient_dir + '/' + 'ts_treatment.csv')
                    ts_is_measured = pd.read_csv(patient_dir + '/' + 'ts_is_measured.csv')
                    static_tasks_binary_multilabel = pd.read_csv(patient_dir + '/' + 'static_tasks_binary_multilabel.csv')
                    final_acuity_outcome = pd.read_csv(patient_dir + '/' + 'Final Acuity Outcome.csv')

                    # here we save full ICU stay in graph, so that we can do random cropping + padding during training also for same graph
                    patient_icuids.append(patient_icuid)
                    patient_vals.append(torch.tensor(ts_vals.values.astype(np.float32)))
                    patient_dems.append(torch.tensor(statics.drop(['ethnicity', 'insurance'], axis=1).values.astype(np.float32)))
                    patient_graph_dems.append(statics[['age', 'gender']])  # can stay df as it will not be used as model input or label
                    patient_treatments.append(torch.tensor(ts_treatment.values.astype(np.float32)))
                    patient_is_measured.append(torch.tensor(ts_is_measured.values.astype(np.float32)))
                    if self.edge_vars in ['half_edge_half_node', 'half_edge_full_node']:
                        patient_half_vals.append(torch.tensor(ts_vals[self.random_features].values.astype(np.float32)))
                        patient_half_is_measured.append(torch.tensor(ts_is_measured[self.random_features_is_measured].values.astype(np.float32)))
                        if self.edge_vars == 'half_edge_half_node':
                            patient_half_vals_node.append(torch.tensor(ts_vals.drop(self.random_features, axis=1).values.astype(np.float32)))
                            patient_half_is_measured_node.append(torch.tensor(ts_is_measured.drop(self.random_features_is_measured, axis=1).values.astype(np.float32)))
                    elif self.edge_vars == 'better_half_corr':
                        patient_half_vals.append(torch.tensor(ts_vals[self.most_correlated_half_features].values.astype(np.float32)))
                        patient_half_is_measured.append(torch.tensor(ts_is_measured[self.most_correlated_half_features_is_measured].values.astype(np.float32)))
                    elif self.edge_vars == 'high_corr':
                        patient_half_vals.append(torch.tensor(ts_vals[self.high_corr_features].values.astype(np.float32)))
                        patient_half_is_measured.append(torch.tensor(ts_is_measured[self.high_corr_features_is_measured].values.astype(np.float32)))

                    y_dict = {}
                    # create tensor for treatments which is one if any element in column is 1
                    y_dict['treatments_bin'] = torch.tensor(np.any(ts_treatment.values, axis=0).astype(np.float32))
                    y_dict['treatments_hour'] = torch.tensor(ts_treatment.values.astype(np.float32))
                    y_dict['vals'] = torch.tensor(
                        ts_vals.values.astype(np.float32))  # for next timestepp task has to be created when cropping is done
                    y_dict['los'] = torch.tensor(static_tasks_binary_multilabel['Long LOS'].values.astype(np.float32))
                    y_dict['rea'] = torch.tensor(static_tasks_binary_multilabel['Readmission 30'].values.astype(np.float32))
                    y_dict['icd'] = torch.tensor(
                        static_tasks_binary_multilabel.drop(['Long LOS', 'Readmission 30'], axis=1).values.astype(np.float32))
                    y_dict['acu'] = torch.tensor(final_acuity_outcome.values.astype(np.float32))
                    patient_ys.append(y_dict)

            train_mask, val_mask, test_mask, train_dev_mask, dev_mask = self.train_val_test_masks(patient_icuids_and_splits)

            if self.edge_vars == 'vals':
                edge_idx, edge_features = self.create_edges_feat_similarity_knn(patient_vals)
            else: #'none'
                edge_idx, edge_features = torch.tensor([], dtype=torch.long), torch.tensor([])

            self.save_data(node_id = torch.tensor(patient_icuids), vals = patient_vals, demographics = torch.stack(patient_dems),
                           treatments = patient_treatments, is_measured = patient_is_measured, edge_idx=edge_idx,
                           edge_features=edge_features, y=patient_ys, final_mask=None, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                           train_dev_mask=train_dev_mask, dev_mask=dev_mask, subset_id=subset_id)


def visualize_graph_mimic(graph, name):
    g = torch_geometric.utils.to_networkx(graph, to_undirected=False, node_attrs=['y'])
    net = Network("900px", "900px", directed=False)
    net.inherit_edge_colors(False)
    # assign color to nodes
    for node, data in g.nodes(data=True):
        if data['y'] == 0:
            color = 'green'
        elif data['y'] == 1:
            color = 'red'

        g.nodes[node]['color'] = color
    net.from_nx(g)
    net.show_buttons()
    net.show(f"mimic_graph_{name}.html")

def visualize_graph_tadpole(graph, name):
    g = torch_geometric.utils.to_networkx(graph, to_undirected=False, node_attrs=['y'])
    net = Network("900px", "900px", directed=False)
    net.inherit_edge_colors(False)
    # assign color to nodes
    for node, data in g.nodes(data=True):
        if data['y'] == 0:
            color = 'green'
        elif data['y'] == 1:
            color = 'red'
        elif data['y'] == 2:
            color = 'blue'

        g.nodes[node]['color'] = color
    net.from_nx(g)
    net.show_buttons()
    net.show(f"tadpole_graph_{name}.html")

if __name__ == '__main__':
    from pyvis.network import Network
    # ds2 = TadpoleDataset(root='data/tadpole', mask=False, name='tadpole', raw_file_name='tadpole_numerical.csv', offset=96, bin_split_idx=5,
    #                     split='val', drop_val_patients=True, cross_val_split=None, fold=0, sim_graph=True, k=5)

    #graph = ds2.get(0)
    # visualize_graph_tadpole(graph, 'k5')

