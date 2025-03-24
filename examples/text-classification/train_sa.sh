#!/bin/bash
SOURCE_LANG=id
LANG_FT=cambridgeltl/xlmr-lang-sft-${SOURCE_LANG}-small

variance_threshold=0.90
keep_ratio=0.05

MODEL_DIR=models/sa/smsa-trimmed/large/svd_pruned_SFT_${SOURCE_LANG}_r_uniform_100_e_100
mkdir -p $MODEL_DIR

python run_text_classification.py \
  --model_name_or_path xlm-roberta-large \
  --train_file data/smsa-trimmed/train.tsv \
  --validation_file data/smsa-trimmed/validation.tsv \
  --output_dir $MODEL_DIR \
  --keep_ratio $keep_ratio \
  --lang_ft $LANG_FT \
  --variance_threshold $variance_threshold \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 10 \
  --sparse_ft_max_epochs_per_iteration 10 \
  --ft_params_num 14155776 \
  --freeze_layer_norm \
  --learning_rate 2e-5 \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --save_steps 500 \
  --eval_metric f1 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2