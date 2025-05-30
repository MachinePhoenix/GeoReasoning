#!/bin/bash

ROOT=/data/yanshi.xy/MLLM_RAFT/Gemma3-Infer/cold_start_fixUnderfitting

epochs=(4 5 6 7 8 9 10 11 12 13 14 15)
# epochs=(1 2 3)

for epoch in "${epochs[@]}"; do
    echo "========================================"
    echo "epoch: $epoch"
    echo "========================================"
    MODEL_ROOT=$ROOT/checkpoint-epoch$epoch

    cd /data/yanshi.xy/MLLM_RAFT/MathVista/evaluation
    CUDA_VISIBLE_DEVICES=2 python gemma_generate_response.py \
    --model gemma3_4b \
    --model_path ${MODEL_ROOT} \
    --output_dir="/data/yanshi.xy/MLLM_RAFT/Gemma3-Infer/gemma_mathvista_test_cold_start" \
    --output_file="output_epoch_$epoch.json"

    echo "Inference stage on MathVista of epoch $epoch completed!"
    
    echo "epoch $epoch completed!"
done