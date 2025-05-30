#!/bin/bash

# 训练参数配置
EPOCHS=5
SAMPLE_NUM=8
TOP_K=1
INITIAL_MODEL="/data/yanshi.xy/hf_models/Qwen2.5-VL-3B-Instruct-SFT"
CHECKPOINT_ROOT="/data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/qwen_models_raft_cw0_dataNoUpdate"
DATA_ROOT="/data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/GeoReasoning/train"
SCRIPTS_PATH="/data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/scripts_raft"
cw=0.0
rw=1.0

# 初始化路径
mkdir -p $CHECKPOINT_ROOT

for ((i=1; i<=$EPOCHS; i++)); do
    echo "================ Epoch $i ================"
    
    if [ $i -eq 1 ]; then
        CURRENT_MODEL=$INITIAL_MODEL
    else
        CURRENT_MODEL="$CHECKPOINT_ROOT/checkpoint-epoch$((i-1))"
    fi

    echo "[1/3] Rollout Phase - Caption Generation:"
    # # 输入数据路径
    INPUT_DATA="$DATA_ROOT/data$((i-1))_cw0_dataNoUpdate.json"
    
    INTER_DATA="$DATA_ROOT/inter_data${i}_cw0_dataNoUpdate.json"
    
    # Caption Generation
    cd /data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/src_raft
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python caption_generation_ray.py \
        --model_path $CURRENT_MODEL \
        --image_path /data/yanshi.xy/MLLM_RL/EasyR1/GeoReasoning/train \
        --data_path $INPUT_DATA \
        --sample_num $SAMPLE_NUM \
        --output_path $INTER_DATA \
        --questions_path /data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/GeoReasoning/questions.json \
        --batch_size 256 \
        --main_gpus 0 1 2 3 4 5 6 7
    
    # 验证中间数据生成
    if [ ! -s $INTER_DATA ]; then
        echo "Error: Caption Generation failed at epoch $i!"
        exit 1
    fi

    echo "[2/3] Rollout Phase - Reasoning Reward:"
    # 最终输出数据路径
    OUTPUT_DATA="$DATA_ROOT/data${i}_cw0_dataNoUpdate.json"
    
    # Reasoning Reward
    cd /data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/src_raft
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python reasoning_rewarding_ray_noStatistics_dataNoUpdate.py \
        --qa_model_path /data/yanshi.xy/hf_models/Qwen2.5-7B-Instruct \
        --original_data_path $INPUT_DATA \
        --intermediate_path $INTER_DATA \
        --output_path $OUTPUT_DATA \
        --caption_weight=$cw \
        --reasoning_weight=$rw \
        --topK $TOP_K \
        --qa_gpus 0 1 2 3 4 5 6 7 \
        --project qwen_raft_cw0_dataNoUpdate \
        --current_epoch $i \
        --run_id_path $SCRIPTS_PATH
        # --run_id qwen_raft_cw0_1sample
    
    # 验证最终数据生成
    if [ ! -s $OUTPUT_DATA ]; then
        echo "Error: Reasoning Reward failed at epoch $i!"
        exit 2
    fi

    ########################################
    # 3. SFT 阶段
    ########################################
    echo "[3/3] SFT Phase:"
    SFT_OUTPUT_DIR="$CHECKPOINT_ROOT/checkpoint-epoch$i"
    
    # 调用SFT脚本（GPU分配调整）
    cd /data/yanshi.xy/MLLM_RAFT/Qwen2.5-VL-Finetune/
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed src/train/train_sft.py \
        --use_liger True \
        --deepspeed scripts/zero3_offload.json \
        --model_id $CURRENT_MODEL \
        --data_path $OUTPUT_DATA \
        --image_folder $DATA_ROOT \
        --remove_unused_columns False \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger False \
        --bf16 True \
        --fp16 False \
        --disable_flash_attn2 False \
        --output_dir $SFT_OUTPUT_DIR \
        --num_train_epochs 1 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --image_min_pixels $((512*28*28)) \
        --image_max_pixels $((1280*28*28)) \
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
        --save_strategy "steps" \
        --save_steps 200 \
        --save_total_limit 10 \
        --dataloader_num_workers 4 \
        --epoch=$i \
        --project_name="RAFT_SFT_epoch${i}_cw0_dataNoUpdate"

    # 验证模型保存
    if [ ! -d "$SFT_OUTPUT_DIR" ]; then
        echo "Error: SFT failed to save model at epoch $i!"
        exit 3
    fi

    echo "Epoch $i completed. Model saved to: $SFT_OUTPUT_DIR"
done