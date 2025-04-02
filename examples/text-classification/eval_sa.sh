#!/bin/bash
LANG=$1

LANG_FT=../language-modeling/models/large/svd_pruned_SFT_${LANG}_r0.90
TASK_FT=models/sa/smsa-trimmed/large/svd_pruned_SFT_id_r_uniform_100_e_100

RESULTS_DIR="results/NusaX-senti/${LANG}"
mkdir -p $RESULTS_DIR

python run_text_classification.py \
  --model_name_or_path xlm-roberta-base \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --dataset_name indonlp/NusaX-senti \
  --dataset_config_name $LANG \
  --output_dir $RESULTS_DIR \
  --do_eval \
  --eval_metric f1 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --eval_split test \
  --overwrite_output_dir 

