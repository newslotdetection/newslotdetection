#!/usr/bin bash

python run.py \
--strategy 'pretrain_init5' \
--labeled_ratio 0.05 \
--select_ratio 0.02 \
--dataset 'atis' \
--method 'ours' \
--setting 'semi_supervised' \
--known_cls_ratio 1.0 \
--seed 0 \
--backbone 'bert_MultiTask' \
--config_file_name 'ours' \
--gpu_id '0' \
--pre_train_multitask \
--save_results \
--save_model \
--results_file_name 'results_ours.csv' \
--alpha 0.05
