#!/bin/bash

port=25000
cfg_path=/home/hdliu/MiniGPT-4/eval_configs/minigptv2_benchmark_evaluation.yaml

CUDA_VISIBLE_DEVICES=1 torchrun --master-port ${port} --nproc_per_node 1 eval_vqa.py \
 --cfg-path ${cfg_path} --dataset cap_decom