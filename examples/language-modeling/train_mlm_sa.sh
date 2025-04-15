#!/bin/bash
LANG=mad

rank_variant=variance
keep_ratio=0.05
variance_threshold=0.90

python run_mlm.py \
  --model_name_or_path xlm-roberta-base \
  --train_file data/sa_corpora/${LANG}.txt \
  --output_dir models/${LANG} \
  --rank_variant $rank_variant \
  --variance_threshold $variance_threshold \
  --keep_ratio $keep_ratio \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 256 \
  --save_steps 1000 \
  --ft_params_num 7667712 \
  --freeze_layer_norm \
  --freeze_decoder \
  --full_l1_reg 0.1 \
  --sparse_l1_reg 0.1 \
  --learning_rate 5e-5 \
  --full_ft_min_steps_per_iteration 10000 \
  --sparse_ft_min_steps_per_iteration 10000 \
  --full_ft_max_steps_per_iteration 100000 \
  --sparse_ft_max_steps_per_iteration 100000 \
  --full_ft_max_epochs_per_iteration 100 \
  --sparse_ft_max_epochs_per_iteration 100 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --validation_split_percentage 5 \
  --load_best_model_at_end \
  --save_total_limit 2 