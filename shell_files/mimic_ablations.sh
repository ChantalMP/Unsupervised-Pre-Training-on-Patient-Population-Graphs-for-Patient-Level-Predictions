#!/bin/sh
# train from scratch models for all rotations and all label ratios
for r in 0 1 2 7 8 9; do
  echo "rotation ${r}"

  for lr in 0.1; do
    # train model
    echo "label ratio ${lr}"
    echo "from scratch"
    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task los --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 1 \
    --peak_lr 1e-4 --end_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 600 --default_root_dir "exps/mimic" \
    --log_every_n_step 10 --check_val_every_n_epoch 2 --edge_vars "vals" --rotation ${r} --pad_mode "pad_emb" --runname "basic_label_ratio${lr}_rot${r}_pt_ablation_mlp_1layer" \
    --label_ratio ${lr} --compute_results --skip_transformer --mlp
    # compute test result
    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task los --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 1 \
    --peak_lr 1e-4 --end_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 600 --default_root_dir "exps/mimic" \
    --log_every_n_step 10 --check_val_every_n_epoch 2 --edge_vars "vals" --rotation ${r} --pad_mode "pad_emb" --runname "test_basic_label_ratio${lr}_rot${r}" \
    --label_ratio ${lr} --test --checkpoint_path "exps/mimic/lightning_logs/basic_label_ratio${lr}_rot${r}_pt_ablation_mlp_1layer/checkpoints/best_auroc.pt" --compute_results \
    --skip_transformer --mlp

    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task los --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 1 \
    --peak_lr 1e-4 --end_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 600 --default_root_dir "exps/mimic" \
    --log_every_n_step 10 --check_val_every_n_epoch 2 --edge_vars "vals" --rotation ${r} --pad_mode "pad_emb" --runname "basic_label_ratio${lr}_rot${r}_pt_ablation_mlp_w_tf_1layer" \
    --label_ratio ${lr} --compute_results --mlp
    # compute test result
    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task los --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 1 \
    --peak_lr 1e-4 --end_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 600 --default_root_dir "exps/mimic" \
    --log_every_n_step 10 --check_val_every_n_epoch 2 --edge_vars "vals" --rotation ${r} --pad_mode "pad_emb" --runname "test_basic_label_ratio${lr}_rot${r}" \
    --label_ratio ${lr} --test --checkpoint_path "exps/mimic/lightning_logs/basic_label_ratio${lr}_rot${r}_pt_ablation_mlp_w_tf_1layer/checkpoints/best_auroc.pt" --compute_results \
    --mlp

    echo "pt_03_24"
    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task los --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.0 --intput_dropout_rate 0.0 --attention_dropout_rate 0.0 --weight_decay 0.0 --n_layers 1 \
    --peak_lr 1e-5 --end_lr 1e-5 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 600 --default_root_dir "exps/mimic" \
    --log_every_n_step 10 --check_val_every_n_epoch 2 --runname "pt_03_24_label_ratio_${lr}_rot${r}_pt_ablation_mlp_1layer" --edge_vars "vals" \
    --rotation ${r} --label_ratio ${lr} --pad_mode "pad_emb" --not_use_dev --skip_transformer --mlp\
    --pretraining_path "exps/mimic_pre_mask/lightning_logs/pt24_mlp_rot${r}/checkpoints/best_score.pt" --compute_results
    # compute test result
    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task los --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.0 --intput_dropout_rate 0.0 --attention_dropout_rate 0.0 --weight_decay 0.0 --n_layers 1 \
    --peak_lr 1e-5 --end_lr 1e-5 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 600 --default_root_dir "exps/mimic" \
    --log_every_n_step 10 --check_val_every_n_epoch 2 --runname "test_pt_03_24_label_ratio_${lr}_rot${r}" --edge_vars "vals" \
    --rotation ${r} --label_ratio ${lr} --pad_mode "pad_emb" --not_use_dev --test --skip_transformer --mlp\
    --checkpoint_path "exps/mimic/lightning_logs/pt_03_24_label_ratio_${lr}_rot${r}_pt_ablation_mlp_1layer/checkpoints/best_auroc.pt" --compute_results

    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task los --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.0 --intput_dropout_rate 0.0 --attention_dropout_rate 0.0 --weight_decay 0.0 --n_layers 1 \
    --peak_lr 1e-5 --end_lr 1e-5 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 600 --default_root_dir "exps/mimic" \
    --log_every_n_step 10 --check_val_every_n_epoch 2 --runname "pt_03_24_label_ratio_${lr}_rot${r}_pt_ablation_mlp_w_tf_1layer" --edge_vars "vals" \
    --rotation ${r} --label_ratio ${lr} --pad_mode "pad_emb" --not_use_dev --mlp\
    --pretraining_path "exps/mimic_pre_mask/lightning_logs/pt24_mlp_with_transformer_rot${r}/checkpoints/best_score.pt" --compute_results
    # compute test result
    python -m graphormer.entry --num_workers 8 --seed 1 --batch_size 5 --dataset_name mimic --task los --gpus 1 --accelerator "ddp" --ffn_dim 256 \
    --hidden_dim 256 --num_heads 8 --dropout_rate 0.0 --intput_dropout_rate 0.0 --attention_dropout_rate 0.0 --weight_decay 0.0 --n_layers 1 \
    --peak_lr 1e-5 --end_lr 1e-5 --edge_type multi_hop --multi_hop_max_dist 20 --warmup_updates 0 --tot_updates 600 --default_root_dir "exps/mimic" \
    --log_every_n_step 10 --check_val_every_n_epoch 2 --runname "test_pt_03_24_label_ratio_${lr}_rot${r}" --edge_vars "vals" \
    --rotation ${r} --label_ratio ${lr} --pad_mode "pad_emb" --not_use_dev --test --mlp\
    --checkpoint_path "exps/mimic/lightning_logs/pt_03_24_label_ratio_${lr}_rot${r}_pt_ablation_mlp_w_tf_1layer/checkpoints/best_auroc.pt" --compute_results
  done

done
