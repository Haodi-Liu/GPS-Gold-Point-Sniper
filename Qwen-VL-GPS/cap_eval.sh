#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use-env \
    --nproc_per_node 1 \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    cap_eval_qwen_decom.py 