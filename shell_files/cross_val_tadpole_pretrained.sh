#!/bin/sh
# finetune models for all rotations and all label ratios

  for lr in 0.01 0.05 0.1 0.5 1.0; do
    # train model
    echo "label ratio ${lr}"
    python -m graphormer.entry --num_workers 0 --seed 1 --batch_size 1 --dataset_name tadpole_class --gpus 1 --accelerator "ddp" --ffn_dim 64 \
    --hidden_dim 64 --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0 --n_layers 4 \
    --peak_lr 5e-6 --end_lr 5e-6 --edge_type multi_hop --multi_hop_max_dist 20 --check_val_every_n_epoch 1 --warmup_updates 0 --tot_updates 1200 \
    --default_root_dir "exps/tadpole_class" --log_every_n_step 1 --runname "cross_val_pretrained_label_ratio${lr}" --cross_val --drop_val_patients \
    --use_sim_graph_tadpole --pretraining_path exps/tadpole_mask/lightning_logs/PT_mask_all_cross_val/checkpoints/epoch6000.pt --label_ratio ${lr} --mask_all

  done
