cd your_path_to_LLaMA-Factory

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train your_path_to_LLaMA-Factory/examples/train_full/gemma3_full_sft.yaml \
    model_name_or_path=google/gemma-3-4b-it \
    freeze_language_model=true \
    output_dir=your_output_dir \
    dataset_dir=your_path_to_GeoReasoning \
    dataset=data_name \
    num_train_epochs=1 \
    per_device_train_batch_size=16 \
    gradient_accumulation_steps=2 \
    learning_rate=1e-5 \
    logging_steps=1 \
    disable_gradient_checkpointing=false \
    flash_attn=fa2
