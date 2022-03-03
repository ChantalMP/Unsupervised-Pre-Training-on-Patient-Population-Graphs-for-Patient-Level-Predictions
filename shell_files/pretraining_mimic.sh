#!/bin/sh

  for fold in 0 1 2 7 8 9; do

    # train model
    echo "fold ${fold}"
    echo "pt24"
    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task pre_mask --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 8 \
    --peak_lr 1e-3 --end_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 3000 --default_root_dir "exps/mimic_pre_mask" \
    --log_every_n_step 10 --check_val_every_n_epoch 5 --runname "pt24_rot${fold}" --edge_vars "vals" --rotation ${fold} --pad_mode "pad_emb" \
    --mask_ratio 0.3 --block_size 24 --not_use_dev

    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task pre_mask --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 8 \
    --peak_lr 1e-3 --end_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 3000 --default_root_dir "exps/mimic_pre_mask" \
    --log_every_n_step 10 --check_val_every_n_epoch 5 --runname "pt24_${fold}" --edge_vars "vals" --rotation ${fold} --pad_mode "pad_emb" \
    --mask_ratio 0.3 --block_size 24 --not_use_dev

    echo "pt6"
    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task pre_mask --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 8 \
    --peak_lr 1e-3 --end_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 3000 --default_root_dir "exps/mimic_pre_mask" \
    --log_every_n_step 10 --check_val_every_n_epoch 5 --runname "pt6_rot${fold}" --edge_vars "vals" --rotation ${fold} --pad_mode "pad_emb" \
    --mask_ratio 1.0 --block_size 6 --not_use_dev

    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task pre_mask --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 8 \
    --peak_lr 1e-3 --end_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 3000 --default_root_dir "exps/mimic_pre_mask" \
    --log_every_n_step 10 --check_val_every_n_epoch 5 --runname "pt6_rot${fold}" --edge_vars "vals" --rotation ${fold} --pad_mode "pad_emb" \
    --mask_ratio 1.0 --block_size 6 --not_use_dev

  done