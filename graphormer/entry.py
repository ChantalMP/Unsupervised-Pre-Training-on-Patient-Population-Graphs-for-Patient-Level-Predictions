# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# file adapted for our pipeline
import json
import multiprocessing
import os
import time
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchinfo import summary

from graphormer.data import GraphDataModule
from graphormer.model import Graphormer


def reset_weights(model):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def init_model(args, cross_val_split=None):
    if args.checkpoint_path != '' and False:
        model = Graphormer.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            gcn=args.gcn,
            mlp=args.mlp,
            skip_transformer=args.skip_transformer,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            use_seq_tokens=args.use_seq_tokens,
            dataset_name=args.dataset_name,
            task=args.task,
            edge_vars=args.edge_vars,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            cross_val_split=cross_val_split,
            label_ratio=args.label_ratio,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
            run_name=args.runname,
            pad_mode=args.pad_mode,
            rotation=args.rotation,
            not_use_dev=args.not_use_dev,
            loss_weighting=args.loss_weighting,
            compute_results=args.compute_results,
            use_sim_graph_tadpole=args.use_sim_graph_tadpole,
            mask_all_tadpole=args.mask_all,
            fold=args.fold
        )
    else:
        model = Graphormer(
            n_layers=args.n_layers,
            gcn=args.gcn,
            mlp=args.mlp,
            skip_transformer=args.skip_transformer,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            use_seq_tokens=args.use_seq_tokens,
            dataset_name=args.dataset_name,
            task=args.task,
            edge_vars=args.edge_vars,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            cross_val_split=cross_val_split,
            label_ratio=args.label_ratio,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
            run_name=args.runname,
            pad_mode=args.pad_mode,
            rotation=args.rotation,
            not_use_dev=args.not_use_dev,
            loss_weighting=args.loss_weighting,
            compute_results=args.compute_results,
            use_sim_graph_tadpole=args.use_sim_graph_tadpole,
            mask_all_tadpole=args.mask_all,
            fold=args.fold
        )

    if args.pretraining_path != '':
        # load model weights except for classification layer
        if args.dataset_name == 'tadpole_class' or args.dataset_name == 'tadpole':
            if cross_val_split == None:
                state_dict = torch.load(args.pretraining_path)
            else:
                path = f"exps/tadpole_mask/lightning_logs/PT_mask_ratio_{args.mask_ratio}_mask_all_fold{cross_val_split}_with_cont_mlp/checkpoints/epoch6000.pt"
                print("Loading pretrained mask_all: ", path)
                state_dict = torch.load(path)
            # delete final layer for fine-tuning
            if args.dataset_name == 'tadpole_class':
                if args.mask_all:
                    state_dict.pop('bin_out_proj_discrete.weight')
                    state_dict.pop('bin_out_proj_discrete.bias')
                    state_dict.pop('bin_out_proj_cont.weight')
                    state_dict.pop('bin_out_proj_cont.bias')
                else:
                    state_dict.pop('bin_out_proj.weight')
                    state_dict.pop('bin_out_proj.bias')

        elif args.dataset_name == 'mimic':
            state_dict = torch.load(args.pretraining_path)
            # delete final layer
            state_dict.pop('bin_out_proj_vals.weight')
            state_dict.pop('bin_out_proj_vals.bias')
            state_dict.pop('bin_out_proj_treat.weight')
            state_dict.pop('bin_out_proj_treat.bias')

        with torch.no_grad():
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
            assert len(missing_keys) == 2, f"Missing keys: {missing_keys}"

        print("Loaded pretrained weights!")

    elif args.checkpoint_path != '':
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')

        with torch.no_grad():
            model.load_state_dict(state_dict, strict=True)
        print("Loaded weights!")

    return model


def cli_main():
    # ------------
    # args
    # ------------

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Graphormer.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    args.replace_sampler_ddp = False
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    if args.compute_results:
        # load df with pandas
        # if args.run_name starts with basic, set name to basic
        if args.runname.startswith('basic') or args.runname.startswith('test_basic_'):
            name = 'basic'
        elif args.runname.startswith('pt_03_24') or args.runname.startswith('test_pt_03_24_'):
            name = 'pt24'
        else:
            name = 'pt6'
        if args.skip_transformer:
            val_df_name = f"final_results/{name}_val_{args.task}_aurocs_pt_ablation_mlp_1l.csv"
            test_df_name = f"final_results/{name}_test_{args.task}_aurocs_pt_ablation_mlp_1l.csv"
            val_df_name_acc = f"final_results/{name}_val_{args.task}_accs_pt_ablation_mlp_1l.csv"
            test_df_name_acc = f"final_results/{name}_test_{args.task}_accs_pt_ablation_mlp_1l.csv"
        else:
            val_df_name = f"final_results/{name}_val_{args.task}_aurocs_pt_ablation_mlp_w_tf_1l.csv"
            test_df_name = f"final_results/{name}_test_{args.task}_aurocs_pt_ablation_mlp_w_tf_1l.csv"
            val_df_name_acc = f"final_results/{name}_val_{args.task}_accs_pt_ablation_mlp_w_tf_1l.csv"
            test_df_name_acc = f"final_results/{name}_test_{args.task}_accs_pt_ablation_mlp_w_tf_1l.csv"
        val_df = pd.read_csv(val_df_name)
        test_df = pd.read_csv(test_df_name)
        val_df_acc = pd.read_csv(val_df_name_acc)
        test_df_acc = pd.read_csv(test_df_name_acc)
        label_ratio = f'{args.label_ratio:.2f}'
        rotation = args.rotation
        if rotation >= 7:
            rotation -= 4
        if args.test:
            if not np.isnan(test_df[label_ratio][rotation]):
                print("WARNING: Result already exists. Skipping this run.")
                return
        else:  # training
            if not np.isnan(val_df[label_ratio][rotation]):
                print("WARNING: Result already exists. Skipping this run.")
                return
            os.makedirs(os.path.dirname(args.default_root_dir + f'/lightning_logs/{args.runname}/checkpoints/'), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(args.default_root_dir + f'/lightning_logs/{args.runname}/checkpoints/'), exist_ok=True)

    if not args.test:
        with open(args.default_root_dir + f'/lightning_logs/{args.runname}/commandline_args.txt', 'w+') as f:
            json.dump(args.__dict__, f, indent=2)

    # ------------
    # data
    # ------------
    if not args.cross_val:  # data has only to be loaded once
        dm = GraphDataModule.from_argparse_args(args)

        # ------------
        # model
        # ------------
        model = init_model(args)
        summary(model)

        if not args.test and not args.validate:
            print(model)

    # ------------
    # training
    # ------------

    dirpath = args.default_root_dir + f'/lightning_logs/{args.runname}/checkpoints'
    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt') and args.checkpoint_path != '':
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)

    if not args.cross_val:  # logging done manually for cross validation, do not save models for all ten folds
        if args.compute_results:
            logger = False
        else:
            logger = TensorBoardLogger(save_dir=args.default_root_dir + f'/lightning_logs', version=f'{args.runname}', name='')
        trainer = pl.Trainer.from_argparse_args(args, logger=logger, checkpoint_callback=False)

        # per default don't save to save space
        if args.save_last:
            checkpoint_callback = ModelCheckpoint(
                monitor=None,
                dirpath=dirpath,
                save_top_k=0,
                save_last=True
            )
            trainer.callbacks.append(checkpoint_callback)
        if not args.compute_results:
            trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

        if args.test:
            result = trainer.test(model, datamodule=dm)
            if not args.compute_results:
                with open(f'test_results_acc.txt', 'a+') as f:
                    f.write(f'{args.runname}: {result[0][f"test_acc_{args.task}"]}\n')
            else:
                # set test df to results
                test_df[label_ratio][rotation] = result[0][f"test_auroc_{args.task}"]
                test_df_acc[label_ratio][rotation] = result[0][f"test_acc_{args.task}"]
                test_df.to_csv(test_df_name, index=False)
                test_df_acc.to_csv(test_df_name_acc, index=False)
            pprint(result)
        elif args.validate:
            result = trainer.validate(model, datamodule=dm, verbose=True)
            if not args.compute_results:
                with open(f'val_results.txt', 'a+') as f:
                    f.write(f'{args.runname}: {result[0][f"val_auroc_{args.task}"]}\n')
            else:
                val_df[label_ratio][rotation] = result[0][f"val_auroc_{args.task}"]
                val_df_acc[label_ratio][rotation] = result[0][f"val_acc_{args.task}"]
                val_df.to_csv(val_df_name, index=False)
                val_df_acc.to_csv(val_df_name_acc, index=False)
            pprint(result)
        else:
            trainer.fit(model, datamodule=dm)
            if args.compute_results:
                val_df[label_ratio][rotation] = trainer.callback_metrics[f'val_best_auroc_{args.task}'].item()
                # overwrite val_df file
                val_df.to_csv(val_df_name, index=False)

    else:  # training with cross validation for TADPOLE
        train_losses = {}
        train_accs = {}
        train_f1s = {}
        train_aurocs = {}
        val_losses = {}
        val_accs = {}
        val_f1s = {}
        val_aurocs = {}
        for fold in range(10):
            pl.seed_everything(args.seed)
            trainer = pl.Trainer.from_argparse_args(args, logger=False)
            dm = GraphDataModule.from_argparse_args(args, cross_val_split=fold, drop_val_patients=args.drop_val_patients, fold=fold)
            model = init_model(args, cross_val_split=fold)
            print(f"Start Training for Fold {fold}")
            trainer.fit(model, datamodule=dm)
            train_losses[fold] = model.train_losses.copy()
            train_accs[fold] = model.train_acc.copy()
            train_f1s[fold] = model.train_f1.copy()
            train_aurocs[fold] = model.train_auroc.copy()
            val_losses[fold] = model.val_losses.copy()
            val_accs[fold] = model.val_acc.copy()
            val_f1s[fold] = model.val_f1.copy()
            val_aurocs[fold] = model.val_auroc.copy()
            model.train_losses = []
            model.train_acc = []
            model.val_losses = []
            model.train_f1 = []
            model.val_f1 = []
            model.val_acc = []
            model.train_auroc = []
            model.val_auroc = []

        logger = TensorBoardLogger(save_dir=args.default_root_dir + f'/lightning_logs', version=f'{args.runname}', name='')

        # metrics per split
        for run_idx, (run_train_losses, run_train_accs, run_val_losses, run_val_accs, run_train_f1s, run_val_f1s, run_train_aurocs, run_val_aurocs) in \
                enumerate(
                    zip(train_losses.values(), train_accs.values(), val_losses.values(), val_accs.values(), train_f1s.values(), val_f1s.values(),
                        train_aurocs.values(), val_aurocs.values())):
            for idx, (train_loss, train_acc, val_loss, train_f1, train_auroc) in enumerate(
                    zip(run_train_losses, run_train_accs, run_val_losses, run_train_f1s, run_train_aurocs)):
                logger.log_metrics(metrics={f'train_loss_run{run_idx}': train_loss, f'train_acc_disease_pred_run{run_idx}': train_acc,
                                            f'train_f1_run{run_idx}': train_f1, f'val_loss_run{run_idx}': val_loss,
                                            f'train_auroc_run{run_idx}': train_auroc}, step=idx)
            for idx, (val_acc, val_f1, val_auroc) in enumerate(zip(run_val_accs, run_val_f1s, run_val_aurocs)):
                logger.log_metrics(metrics={f'val_acc_disease_pred_run{run_idx}': val_acc, f'val_f1_run{run_idx}': val_f1,
                                            f'val_auroc_disease_pred_run{run_idx}': val_auroc}, step=idx * 10)

        # metrics averaged over splits
        avg_train_losses = [sum(x) / len(x) for x in zip(*train_losses.values())]
        avg_train_acc = [sum(x) / len(x) for x in zip(*train_accs.values())]
        avg_train_f1 = [sum(x) / len(x) for x in zip(*train_f1s.values())]
        avg_train_aurocs = [sum(x) / len(x) for x in zip(*train_aurocs.values())]
        avg_val_losses = [sum(x) / len(x) for x in zip(*val_losses.values())]
        avg_val_acc = [sum(x) / len(x) for x in zip(*val_accs.values())]
        avg_val_f1 = [sum(x) / len(x) for x in zip(*val_f1s.values())]
        avg_val_aurocs = [sum(x) / len(x) for x in zip(*val_aurocs.values())]

        # push plots to tensorboard
        for idx, (train_loss, train_acc, val_loss, train_f1, train_auroc) in enumerate(
                zip(avg_train_losses, avg_train_acc, avg_val_losses, avg_train_f1, avg_train_aurocs)):
            logger.log_metrics(metrics={'train_loss': train_loss, 'train_acc_disease_pred': train_acc, 'train_f1': train_f1, 'val_loss': val_loss,
                                        'train_auroc': train_auroc}, step=idx)
        for idx, (val_acc, val_f1, val_auroc) in enumerate(zip(avg_val_acc, avg_val_f1, avg_val_aurocs)):
            logger.log_metrics(metrics={'val_acc_disease_pred': val_acc, 'val_f1': val_f1, 'val_auroc': val_auroc}, step=(idx + 1) * 10)
        time.sleep(3)

        f = open("last_epoch_avg_results.txt", "a+")
        f.write(f'{args.runname}:\n')
        # index for which the average of val_accs was highest
        f.write(f'val acc avg: {avg_val_acc[-1]}\n')
        f.write(f'val loss avg: {avg_val_losses[-1]}\n')  # zeroth epoch is logged twice because of pipeline sanity check
        f.write(f'val f1 avg: {avg_val_f1[-1]}\n')
        f.write(f'val auroc avg: {avg_val_aurocs[-1]}\n')
        f.write(f'train loss avg: {avg_train_losses[-1]}\n')
        f.write(f'train acc avg: {avg_train_acc[-1]}\n')
        f.write(f'train f1 avg: {avg_train_f1[-1]}\n')
        f.write(f'train auroc avg: {avg_train_aurocs[-1]}\n')
        f.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    cli_main()
