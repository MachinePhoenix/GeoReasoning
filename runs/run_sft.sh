# CUDA_VISIBLE_DEVICES=2,3,4,6 llamafactory-cli train /data/yanshi.xy/LLaMA-Factory/examples/train_full/qwen25vl_full_sft.yaml \
#     num_train_epochs=4 \
#     learning_rate=1e-5 \
#     logging_steps=1 \
#     disable_gradient_checkpointing=False \
#     flash_attn=fa2 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train /data/yanshi.xy/LLaMA-Factory/examples/train_full/qwen3_full_sft.yaml \
    num_train_epochs=5 \
    learning_rate=1e-5 \
    logging_steps=1 \
    disable_gradient_checkpointing=False \
    flash_attn=fa2 


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train /data/yanshi.xy/LLaMA-Factory/examples/train_full/gemma3_full_sft.yaml \
#     model_name_or_path=/data/yanshi.xy/hf_models/Gemma3-4B \
#     output_dir=/data/yanshi.xy/MLLM_RAFT/Gemma3-Infer/sft_4epochs_lr-5 \
#     dataset=geo_train \
#     num_train_epochs=4 \
#     learning_rate=1e-5 \
#     logging_steps=1 \
#     disable_gradient_checkpointing=False \
#     flash_attn=fa2 


#todo debugging
# CUDA_VISIBLE_DEVICES=2 python -m llamafactory.cli train /data/yanshi.xy/LLaMA-Factory/examples/train_full/gemma3_full_sft.yaml \
#     num_train_epochs=4 \
#     learning_rate=1e-5 \
#     logging_steps=1 \
#     disable_gradient_checkpointing=False \
#     flash_attn=fa2 