#!/bin/bash

# 参数验证
# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <new_data_path>"
#     exit 1
# fi
# new_data_path=$1

# 动态获取路径（通过环境变量）
MODEL_NAME="/data/yanshi.xy/hf_models/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/cold_start_fixUnderfitting"
data_path=/data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/GeoReasoning/train/data.json

# 训练参数配置
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=32
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id "$MODEL_NAME" \
    --data_path "$data_path" \
    --image_folder /data/yanshi.xy/MLLM_RL/EasyR1/GeoReasoning/train \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 10 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_steps 200 \
    --save_total_limit 11 \
    --dataloader_num_workers 4 \
    --project_name qwen_cold_start_fixUnderfitting