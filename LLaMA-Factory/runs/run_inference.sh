# CUDA_VISIBLE_DEVICES=2 python scripts/caption_generation_llamafactory.py \
#     --model_name_or_path /data/yanshi.xy/hf_models/Qwen2.5-VL-3B-Instruct \
#     --template qwen2_vl \
#     --infer_backend vllm \
#     --dataset geo_train \
#     --data_path /data/yanshi.xy/LLaMA-Factory/data/geo_train.json \
#     --questions_path /data/yanshi.xy/MLLM_RL/EasyR1/GeoReasoning/questions.json \
#     --sample_num 8


# CUDA_VISIBLE_DEVICES=4 python scripts/caption_generation_llamafactory.py \
#     --model_name_or_path /data/yanshi.xy/hf_models/Gemma3-4B \
#     --template gemma3 \
#     --infer_backend vllm \
#     --dataset geo_train \
#     --data_path /data/yanshi.xy/LLaMA-Factory/data/geo_train.json \
#     --questions_path /data/yanshi.xy/MLLM_RL/EasyR1/GeoReasoning/questions.json \
#     --sample_num 8

# CUDA_VISIBLE_DEVICES=4 python scripts/reasoning_reward_llamafactory.py \
#     --model_name_or_path /data/yanshi.xy/hf_models/Qwen2.5-7B-Instruct \
#     --template qwen \
#     --dataset alpaca_en_demo

# python scripts/eval_bleu_rouge.py generated_predictions.jsonl