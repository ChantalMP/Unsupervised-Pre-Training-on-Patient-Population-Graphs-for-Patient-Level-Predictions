#!/bin/sh

  mask_ratio=0.3
  # changed to pre-train mlp model
  echo "mask_ratio ${mask_ratio}"
  for fold in 0 1 2 3 4 5 6 7 8 9; do
    # train model
    echo "fold ${fold}"
    echo "pretraining"
    python -m graphormer.entry --num_workers 0 --seed 1 --batch_size 1 --dataset_name tadpole --gpus 1 --accelerator "ddp" --ffn_dim 64 --hidden_dim 64 \
    --num_heads 8 --dropout_rate 0.4 --intput_dropout_rate 0.4 --attention_dropout_rate 0.4 --weight_decay 0.0 --n_layers 4 --peak_lr 1e-5 --end_lr 1e-5 \
    --edge_type multi_hop --multi_hop_max_dist 20 --check_val_every_n_epoch 1 --warmup_updates 0 --tot_updates 6000 --default_root_dir "exps/tadpole_mask"\
    --log_every_n_step 1 --runname "PT_mask_ratio_${mask_ratio}_mask_all_fold${fold}_with_cont" --fold ${fold} --drop_val_patients --use_sim_graph_tadpole \
    --not_use_dev --mask_all --mask_ratio ${mask_ratio}

  done
