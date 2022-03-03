# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# code was adapted for our datasets

from functools import partial

from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as DataLoader_geo

from graphormer.collator import collator
from graphormer.utils.evaluators import FeatAccuracy
from graphormer.wrapper import MyTadpoleDataset, MyMIMICDataset


def get_dataset(dataset_name='abaaba', cross_val_split=None, drop_val_patients=None, fold=None, task=None, num_graphs=43,
                edge_vars='full_edge_full_node', gcn=False, mlp=False, rotation=0, pad_mode='original', mask_ratio=0.15, block_size=6, sim_graph=True,
                mask_all=False, k=5):
    global dataset
    global cross_val_datasets
    if dataset is not None and cross_val_split is None:
        return dataset
    elif cross_val_split in cross_val_datasets:  # need to load new dataset for this split
        return cross_val_datasets[cross_val_split]

    elif dataset_name == 'tadpole':
        mask = True if dataset_name == 'tadpole' else False
        dataset = {
            'num_class': None,  # need to treat binary and regression features separately
            'loss_fn': F.cross_entropy,
            'metric': 'macro_acc_avg',
            'metric_mode': 'max',
            'evaluator': FeatAccuracy(),
            'train_dataset': MyTadpoleDataset(root='data/tadpole', mask=mask, name='tadpole', raw_file_name='tadpole_numerical.csv', offset=96,
                                              bin_split_idx=5, split='train', drop_val_patients=drop_val_patients, cross_val_split=cross_val_split,
                                              fold=fold, sim_graph=sim_graph, mask_all=mask_all, mask_ratio=mask_ratio),
            'valid_dataset': MyTadpoleDataset(root='data/tadpole', mask=mask, name='tadpole', raw_file_name='tadpole_numerical.csv', offset=96,
                                              bin_split_idx=5, split='val', drop_val_patients=drop_val_patients, cross_val_split=cross_val_split,
                                              fold=fold, sim_graph=sim_graph, mask_all=mask_all, mask_ratio=mask_ratio),
            'update_mask': None,
            'max_node': 565,
        }
        if cross_val_split is not None:
            cross_val_datasets[cross_val_split] = dataset

    elif dataset_name == 'tadpole_class':
        mask = True if dataset_name == 'tadpole' else False
        dataset = {
            'num_class': None,  # need to treat binary and regression features separately
            'loss_fn': F.cross_entropy,
            'metric': 'macro_acc_avg',
            'metric_mode': 'max',
            'evaluator': FeatAccuracy(),
            'train_dataset': MyTadpoleDataset(root='data/tadpole', mask=mask, name='tadpole', raw_file_name='tadpole_numerical.csv', offset=96,
                                              bin_split_idx=5, split='train', drop_val_patients=drop_val_patients, cross_val_split=cross_val_split,
                                              fold=fold, sim_graph=sim_graph, k=k),
            'valid_dataset': MyTadpoleDataset(root='data/tadpole', mask=mask, name='tadpole', raw_file_name='tadpole_numerical.csv', offset=96,
                                              bin_split_idx=5, split='val', drop_val_patients=drop_val_patients, cross_val_split=cross_val_split,
                                              fold=fold, sim_graph=sim_graph, k=k),
            'update_mask': None,
            'max_node': 565,
        }
        if cross_val_split is not None:
            cross_val_datasets[cross_val_split] = dataset

    elif dataset_name == 'mimic':
        if task.startswith('pre'):
            dataset = {
                'num_class': None,  # need to treat binary and regression features separately
                'loss_fn': F.cross_entropy,
                'metric': 'macro_acc_avg',
                'metric_mode': 'max',
                'evaluator': FeatAccuracy(),
                'train_dataset': MyMIMICDataset(root=f'data/mimic-iii-0', drop_val_patients=True, use_treatment_input=True, task=task, split='train',
                                                edge_vars=edge_vars, num_graphs=num_graphs, gcn=gcn, mlp=mlp, rotation=rotation, pad_mode=pad_mode,
                                                mask_ratio=mask_ratio, block_size=block_size),
                'valid_dataset': MyMIMICDataset(root=f'data/mimic-iii-0', drop_val_patients=True, use_treatment_input=True, task=task, split='val',
                                                edge_vars=edge_vars, num_graphs=num_graphs, gcn=gcn, mlp=mlp, rotation=rotation, pad_mode=pad_mode,
                                                mask_ratio=mask_ratio, block_size=block_size),
                'test_dataset': MyMIMICDataset(root=f'data/mimic-iii-0', drop_val_patients=True, use_treatment_input=True, task=task, split='test',
                                               edge_vars=edge_vars, num_graphs=num_graphs, gcn=gcn, mlp=mlp, rotation=rotation, pad_mode=pad_mode,
                                               mask_ratio=mask_ratio, block_size=block_size),
                'predict_dataset': MyMIMICDataset(root=f'data/mimic-iii-0', drop_val_patients=True, use_treatment_input=True, task=task, split='val',
                                                  edge_vars=edge_vars, num_graphs=num_graphs, gcn=gcn, mlp=mlp, rotation=rotation, predict=True,
                                                  pad_mode=pad_mode, mask_ratio=mask_ratio, block_size=block_size),
                'update_mask': None,
                'max_node': 550,
            }
        else:
            dataset = {
                'num_class': None,  # need to treat binary and regression features separately
                'loss_fn': F.cross_entropy,
                'metric': 'macro_acc_avg',
                'metric_mode': 'max',
                'evaluator': FeatAccuracy(),
                'train_dataset': MyMIMICDataset(root=f'data/mimic-iii-0', drop_val_patients=True, use_treatment_input=True, task=task, split='train',
                                                edge_vars=edge_vars, num_graphs=num_graphs, gcn=gcn, mlp=mlp, rotation=rotation, pad_mode=pad_mode,
                                                k=k),
                'valid_dataset': MyMIMICDataset(root=f'data/mimic-iii-0', drop_val_patients=True, use_treatment_input=True, task=task, split='val',
                                                edge_vars=edge_vars, num_graphs=num_graphs, gcn=gcn, mlp=mlp, rotation=rotation, pad_mode=pad_mode,
                                                k=k),
                'test_dataset': MyMIMICDataset(root=f'data/mimic-iii-0', drop_val_patients=True, use_treatment_input=True, task=task, split='test',
                                               edge_vars=edge_vars, num_graphs=num_graphs, gcn=gcn, mlp=mlp, rotation=rotation, pad_mode=pad_mode,
                                               k=k),
                'update_mask': None,
                'max_node': 550,
            }

    else:
        raise NotImplementedError

    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset


class GraphDataModule(LightningDataModule):
    name = "OGB-GRAPH"

    def __init__(
            self,
            dataset_name: str = 'ogbg-molpcba',
            cross_val_split=None,
            drop_val_patients=True,
            fold=None,
            task=None,
            num_graphs=None,
            rotation=0,
            edge_vars='full_edge_full_node',
            num_workers: int = 0,
            batch_size: int = 256,
            seed: int = 42,
            multi_hop_max_dist: int = 5,
            spatial_pos_max: int = 1024,
            gcn=False,
            mlp=False,
            pad_mode='original',
            mask_ratio=0.15,
            block_size=6,
            sim_graph=True,
            mask_all=False,
            k=5,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name, cross_val_split=cross_val_split, drop_val_patients=drop_val_patients, fold=fold, task=task,
                                   num_graphs=num_graphs, edge_vars=edge_vars, gcn=gcn, mlp=mlp, rotation=rotation, pad_mode=pad_mode,
                                   mask_ratio=mask_ratio, block_size=block_size, mask_all=mask_all, sim_graph=sim_graph, k=k)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.gcn = gcn
        self.mlp = mlp
        self.task = task
        self.pad_mode = pad_mode

    def setup(self, stage: str = None):
        self.dataset_train = self.dataset['train_dataset']
        self.dataset_val = self.dataset['valid_dataset']
        if 'test_dataset' in self.dataset.keys():
            self.dataset_test = self.dataset['test_dataset']
        if 'predict_dataset' in self.dataset.keys():
            self.dataset_predict = self.dataset['predict_dataset']
        self.collator = collator

    def train_dataloader(self):
        if self.gcn or self.mlp:
            if self.task == "pre_mask":
                loader = DataLoader(
                    self.dataset_train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=partial(self.collator, dataset=self.dataset_name, max_node=get_dataset(self.dataset_name)[
                        'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, gcn=self.gcn or self.mlp,
                                       pad_mode=self.pad_mode),
                )
            else:
                loader = DataLoader_geo(
                    dataset=self.dataset_train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                )
        else:
            loader = DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(self.collator, dataset=self.dataset_name, max_node=get_dataset(self.dataset_name)[
                    'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, gcn=self.gcn,
                                   pad_mode=self.pad_mode),
            )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        if self.gcn or self.mlp:
            if self.task == "pre_mask":
                loader = DataLoader(
                    self.dataset_val,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=partial(self.collator, dataset=self.dataset_name, max_node=get_dataset(self.dataset_name)[
                        'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, gcn=self.gcn or self.mlp,
                                       pad_mode=self.pad_mode),
                )
            else:
                loader = DataLoader_geo(
                    dataset=self.dataset_val,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
        else:
            loader = DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=partial(self.collator, dataset=self.dataset_name, max_node=get_dataset(self.dataset_name)[
                    'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, gcn=self.gcn,
                                   pad_mode=self.pad_mode),
            )
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader(self):
        if self.gcn or self.mlp:
            if self.task == "pre_mask":
                loader = DataLoader(
                    self.dataset_test,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=partial(self.collator, dataset=self.dataset_name, max_node=get_dataset(self.dataset_name)[
                        'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, gcn=self.gcn or self.mlp,
                                       pad_mode=self.pad_mode),
                )
            else:
                loader = DataLoader_geo(
                    dataset=self.dataset_test,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
        else:
            loader = DataLoader(
                self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=partial(self.collator, dataset=self.dataset_name, max_node=get_dataset(self.dataset_name)[
                    'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, gcn=self.gcn,
                                   pad_mode=self.pad_mode),
            )
        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self.dataset_predict,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(self.collator, dataset=self.dataset_name, max_node=get_dataset(self.dataset_name)[
                'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, pad_mode=self.pad_mode),
        )
        return loader
