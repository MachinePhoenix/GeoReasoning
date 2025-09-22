#!/bin/bash

EPOCHS=4
SAMPLE_NUM=8
TOP_K=1
sft_epoch=1
INITIAL_MODEL=your_initial_model_path
rewarding_model=qwen2.5_7b
project_name=gemma_raft_data_rw07_cs1_sft${sft_epoch}_${rewarding_model}
CHECKPOINT_ROOT=your_path_to_checkpoints/$project_name
DATA_ROOT=your_path_to_GeoReasoning
DATA_INFO_PATH=your_path_to_GeoReasoning/dataset_info.json
cw=0.3
rw=0.7
correct_weight=0.9
run_id_path=your_path_to_run_ids/$project_name
total_len=10000
step_size=10000
STEPS=$(($total_len / $step_size))

if [ ! -d "$CHECKPOINT_ROOT" ]; then
    mkdir -p "$CHECKPOINT_ROOT"
    echo "Path does not exist, mkdir $CHECKPOINT_ROOT"
else
    echo "Path already exists at: $CHECKPOINT_ROOT"
fi

if [ ! -d "$run_id_path" ]; then
    mkdir -p "$run_id_path"
    echo "Path does not exist, mkdir $run_id_path"
else
    echo "Path already exists at: $run_id_path"
fi

mkdir -p $CHECKPOINT_ROOT
export RAY_memory_usage_threshold=1.0
for ((i=0; i<=$EPOCHS; i++)); do
    echo "================ Epoch $i ================"

    data_name=data${i}
    new_data_name=data$((i+1))
    INPUT_DATA="$DATA_ROOT/$data_name.json"
    INTER_DATA="$DATA_ROOT/inter_data${i}.json"

    if [ $i -eq 0 ]; then
        cp $DATA_ROOT/data.json $INPUT_DATA
    fi

    cd ../src_raft
    python add_data_item.py --data_name $data_name --data_info_path $DATA_INFO_PATH
    python add_data_item.py --data_name $new_data_name --data_info_path $DATA_INFO_PATH

    if [ $i -eq 0 ]; then
        cd  ../src_raft
        echo "[0/3] Data preprocessing - Add a new item of to save the original ground truth caption: "
        python data0_preprocessing.py --file_path=$INPUT_DATA
    fi

    echo "[0.5/3] Shuffle dataset:"
    cd /storage1/jiaxinh/Active/yuexin/MLLM_RAFT/Gemma3-Infer/src_raft
    python shuffle_data.py --input_data $INPUT_DATA

    for ((j=0; j<$STEPS; j++)); do
        echo "================ Step $j of Epoch $i ================"

        if [ $i -eq 0 ] && [ $j -eq 0 ]; then
            CURRENT_MODEL="$INITIAL_MODEL"
        else
            CURRENT_MODEL="$CHECKPOINT_ROOT/checkpoint-epoch${i}-step${j}"
        fi

        echo "[1/3] Rollout Phase - Caption Generation:"
        cd your_path_to_LLaMA-Factory
        export RAY_memory_usage_threshold=1.0
        CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src_raft/caption_generation_llamafactory_ray.py \
            --model_name_or_path $CURRENT_MODEL \
            --template gemma3 \
            --dataset $data_name \
            --data_path $INPUT_DATA \
            --save_path $INTER_DATA \
            --sample_num $SAMPLE_NUM \
            --step $j \
            --step_size $step_size
    
        
        if [ ! -s $INTER_DATA ]; then
            echo "Error: Caption Generation failed at epoch $i!"
            exit 1
        fi

        echo "[2/3] Rollout Phase - Reasoning Reward:"
        OUTPUT_DATA="$DATA_ROOT/$new_data_name.json"

        # Reasoning Rewarding
        cd ../src_raft
        CUDA_VISIBLE_DEVICES=0,1,2,3 python reasoning_rewarding_ray_noStatistics_bystep_updateBest.py \
            --qa_model_path "Qwen/Qwen2.5-7B-Instruct" \
            --original_data_path $INPUT_DATA \
            --intermediate_path $INTER_DATA \
            --output_path $OUTPUT_DATA \
            --caption_weight=$cw \
            --reasoning_weight=$rw \
            --correct_weight=$correct_weight \
            --topK $TOP_K \
            --project $project_name \
            --current_epoch $i \
            --run_id_path $run_id_path \
            --step $j \
            --step_size $step_size \
            --step_num $STEPS \
        
        if [ ! -s $OUTPUT_DATA ]; then
            echo "Error: Reasoning Reward failed at step $j of epoch $i!"
            exit 2
        fi

        echo "[3/3] SFT Phase:"
        if [ $j -lt $(($STEPS-1)) ]; then
            SFT_OUTPUT_DIR="$CHECKPOINT_ROOT/checkpoint-epoch${i}-step$((j+1))"
        else
            SFT_OUTPUT_DIR="$CHECKPOINT_ROOT/checkpoint-epoch$((i+1))-step0"
        fi

        cd your_path_to_LLaMA-Factory
        FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train your_path_to_LLaMA-Factory/examples/train_full/gemma3_full_sft.yaml \
            model_name_or_path=$CURRENT_MODEL \
            freeze_language_model=true \
            output_dir=$SFT_OUTPUT_DIR \
            dataset=$new_data_name \
            num_train_epochs=1 \
            per_device_train_batch_size=8 \
            gradient_accumulation_steps=4 \
            learning_rate=1e-5 \
            logging_steps=1 \
            disable_gradient_checkpointing=false \
            flash_attn=fa2

        if [ ! -d "$SFT_OUTPUT_DIR" ]; then
            echo "Error: SFT failed to save model at Step $j of Epoch $i!"
            exit 3
        fi
        echo "Step $j of Epoch $i completed. Model saved to: $SFT_OUTPUT_DIR"
    done
done


