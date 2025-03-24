#!/bin/bash
LANG=$1 

LANG_FT=../language-modeling/models/svd_pruned_SFT_${LANG}_r_uniform_200_e_200
TASK_FT=models/nli/svd_pruned_SFT_en_r_0.90_without_source_lang_initialization

RESULTS_DIR=results/AmericasNLI/${LANG}
mkdir -p $RESULTS_DIR

python run_text_classification.py \
  --model_name_or_path xlm-roberta-base \
  --test_file data/AmericasNLI/test/${LANG}.tsv \
  --input_columns premise hypothesis \
  --label_file data/anli_labels.json \
  --output_dir $RESULTS_DIR \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --do_eval \
  --eval_split test \
  --eval_metric xnli \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir 