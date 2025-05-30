#!/bin/bash

# 训练参数配置
EPOCHS=5
SAMPLE_NUM=8
TOP_K=1
INITIAL_MODEL="/data/yanshi.xy/hf_models/Gemma3-4B-SFT"
CHECKPOINT_ROOT="/data/yanshi.xy/MLLM_RAFT/Gemma3-Infer/gemma_models_raft_rw0"
DATA_ROOT="/data/yanshi.xy/LLaMA-Factory/data"
cw=1.0
rw=0.0

# 初始化路径
mkdir -p $CHECKPOINT_ROOT

for ((i=3; i<=$EPOCHS; i++)); do
    echo "================ Epoch $i ================"
    
    if [ $i -eq 1 ]; then
        CURRENT_MODEL=$INITIAL_MODEL
    else
        CURRENT_MODEL="$CHECKPOINT_ROOT/checkpoint-epoch$((i-1))"
    fi

    # echo "[1/3] Rollout Phase - Caption Generation:"
    # # # 输入数据路径
    data_name=geo_train_gemma3_$((i-1))_rw0
    INPUT_DATA="$DATA_ROOT/$data_name.json"
    INTER_DATA="$DATA_ROOT/gemma3_inter_data${i}_rw0.json"
    
    # Caption Generation
    cd /data/yanshi.xy/LLaMA-Factory
    # export RAY_memory_monitor_refresh_ms=0
    export RAY_memory_usage_threshold=1.0
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/caption_generation_llamafactory_ray.py \
        --model_name_or_path $CURRENT_MODEL \
        --template gemma3 \
        --dataset $data_name \
        --data_path $INPUT_DATA \
        --questions_path /data/yanshi.xy/MLLM_RL/EasyR1/GeoReasoning/questions.json \
        --save_path $INTER_DATA \
        --sample_num 8
    
    # 验证中间数据生成
    if [ ! -s $INTER_DATA ]; then
        echo "Error: Caption Generation failed at epoch $i!"
        exit 1
    fi

    echo "[2/3] Rollout Phase - Reasoning Reward:"
    # # 最终输出数据路径
    OUTPUT_DATA="$DATA_ROOT/geo_train_gemma3_${i}_rw0.json"
    
    # Reasoning Reward
    cd /data/yanshi.xy/MLLM_RAFT/Gemma3-Infer/src_raft
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python reasoning_rewarding_ray_noStatistics.py \
        --qa_model_path /data/yanshi.xy/hf_models/Qwen2.5-7B-Instruct \
        --original_data_path $INPUT_DATA \
        --intermediate_path $INTER_DATA \
        --output_path $OUTPUT_DATA \
        --caption_weight=$cw \
        --reasoning_weight=$rw \
        --topK $TOP_K \
        --qa_gpus 0 1 2 3 4 5 6 7
    
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
    
    cd /data/yanshi.xy/LLaMA-Factory
    # Step 3: 模型训练
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train /data/yanshi.xy/LLaMA-Factory/examples/train_full/gemma3_full_sft.yaml \
        model_name_or_path=$CURRENT_MODEL \
        output_dir=$CHECKPOINT_ROOT/checkpoint-epoch$i \
        dataset=geo_train_gemma3_${i}_rw0 \
        num_train_epochs=1 \
        per_device_train_batch_size=4 \
        gradient_accumulation_steps=8 \
        learning_rate=1e-5 \
        logging_steps=1 \
        disable_gradient_checkpointing=false \
        flash_attn=fa2

    # 验证模型保存
    if [ ! -d "$SFT_OUTPUT_DIR" ]; then
        echo "Error: SFT failed to save model at epoch $i!"
        exit 3
    fi

    echo "Epoch $i completed. Model saved to: $SFT_OUTPUT_DIR"
done