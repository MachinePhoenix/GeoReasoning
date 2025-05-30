#!/bin/bash

ROOT=/data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/qwen_models_raft_cw0

epochs=(1 2 3 4 5 6)


for epoch in "${epochs[@]}"; do
    echo "========================================"
    echo "epoch: $epoch"
    echo "========================================"
    MODEL_ROOT=$ROOT/checkpoint-epoch$epoch

    cd /data/yanshi.xy/MLLM_RAFT/MathVista/evaluation
    CUDA_VISIBLE_DEVICES=6 python qwen_generate_response.py \
    --model qwen2.5_vl_3b \
    --model_path ${MODEL_ROOT} \
    --output_dir="/data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/qwen_mathvista_cw0" \
    --output_file="output_epoch_$epoch.json"

    echo "Inference stage on MathVista of epoch $epoch completed!"
    
    echo "epoch $epoch completed!"
done