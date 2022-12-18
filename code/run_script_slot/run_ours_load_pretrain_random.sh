#!/usr/bin bash

python run.py \
--strategy 'RandomSampling' \
--select_ratio 0.02 \
--dataset 'atis' \
--method 'ours' \
--setting 'semi_supervised' \
--known_cls_ratio 1.0 \
--labeled_ratio 0.05 \
--seed 0 \
--backbone 'bert_MultiTask' \
--config_file_name 'ours' \
--gpu_id '0' \
--train_multitask \
--thr 0.9 \
--save_results \
--save_model \
--results_file_name 'results_ours.csv' \
--fine_tune_epoch 2 \
--alpha 0.05
