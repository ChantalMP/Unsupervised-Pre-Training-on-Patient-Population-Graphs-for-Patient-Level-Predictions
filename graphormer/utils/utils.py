import json
import os

import numpy as np
import torch


# create 10 folds for cross-validation on TADPOLE
def generate_stratified_cross_val_split(fold=0):
    data = torch.load(f'data/tadpole/processed/tadpole_graph_class_fold{fold}.pt')  # all data
    train_idxs = data.node_id
    labels_train = data.y
    labels0 = np.array([idx for idx, label in zip(train_idxs, labels_train) if label == 0])
    labels1 = np.array([idx for idx, label in zip(train_idxs, labels_train) if label == 1])
    labels2 = np.array([idx for idx, label in zip(train_idxs, labels_train) if label == 2])
    np.random.shuffle(labels0)
    np.random.shuffle(labels1)
    np.random.shuffle(labels2)

    splits0 = np.array_split(labels0, 10)
    splits1 = np.array_split(labels1, 10)
    splits2 = np.array_split(labels2, 10)

    for idx, (split0, split1, split2) in enumerate(zip(splits0, splits1, splits2)):
        val_split = np.concatenate([split0, split1, split2])
        np.save(f'data/tadpole/split/cross_val/val_idxs_fold{idx}_strat.npy', val_split)
        rest = np.array([idx for idx in train_idxs if idx not in val_split])
        np.save(f'data/tadpole/split/cross_val/train_idxs_fold{idx}_strat.npy', rest)


# randomly generate indices to drop for simulating missing label case on TADPOLE
def generate_labels_to_drop_balanced_tadpole(fold):
    data = torch.load(f'data/tadpole/processed/tadpole_graph_class_drop_val_train_fold{fold}_sim.pt')
    train_idxs = data.node_id
    labels_train = data.y
    labels0 = [idx for idx, label in zip(train_idxs, labels_train) if label == 0]
    labels1 = [idx for idx, label in zip(train_idxs, labels_train) if label == 1]
    labels2 = [idx for idx, label in zip(train_idxs, labels_train) if label == 2]

    for ratio in [0.01, 0.05, 0.1, 0.5]:
        drop0 = np.random.choice(labels0, size=round(len(labels0) * (1 - ratio)), replace=False)
        drop1 = np.random.choice(labels1, size=round(len(labels1) * (1 - ratio)), replace=False)
        drop2 = np.random.choice(labels2, size=round(len(labels2) * (1 - ratio)), replace=False)
        drop = np.concatenate([drop0, drop1, drop2])
        np.save(f'data/tadpole/split/label_drop_idxs_fold{fold}_{ratio}_bal_sim.npy', drop)


# randomly generate indices to drop for simulating missing label case on MIMIC-III
def generate_labels_to_drop_mimic(rotation):
    # currently for los
    with open(f'data/mimic-iii-0/rotations/train_patients_rot_{rotation}.json') as f:
        train_idxs = json.load(f)

    for ratio in [0.01, 0.1, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        label_idxs = np.random.choice(train_idxs, size=round(len(train_idxs) * (1 - ratio)), replace=False)
        # if folders are not created, create them
        if not os.path.exists(f'data/mimic-iii-0/drop/los'):
            os.makedirs(f'data/mimic-iii-0/drop/los')
        np.save(f'data/mimic-iii-0/drop/los/label_drop_idxs_rot{rotation}_{ratio}.npy', label_idxs)


if __name__ == '__main__':
    pass
