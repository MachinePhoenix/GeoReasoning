
MODEL_BASE=/data/yanshi.xy/MLLM_RAFT/Gemma3-Infer/gemma_models_raft_cw0
OUTPUT_DIR=/data/yanshi.xy/MLLM_RAFT/Gemma3-Infer/gemma_mathverse_cw0
IMAGE_FOLDER=/data/yanshi.xy/MLLM_RAFT/MathVerse/MathVerse/images
QUESTION_FILE=/data/yanshi.xy/MLLM_RAFT/MathVerse/MathVerse/question_eval_all.json
GT_FILE=/data/yanshi.xy/MLLM_RAFT/MathVerse/MathVerse/testmini.json
cd /data/yanshi.xy/MLLM_RAFT/MathVerse
for i in {1..6};do
    MODEL_PATH="$MODEL_BASE/checkpoint-epoch$i"
    ANSWER_FILE="$OUTPUT_DIR/infer_raft_model$i.jsonl"
    EXTRACTION_FILE="$OUTPUT_DIR/extraction_raft_model$i.json"
    SCORE_FILE="$OUTPUT_DIR/score_raft_model$i.json"

    LOG_INFER="$OUTPUT_DIR/log_infer_raft_model$i.log"
    LOG_EXTRACT="$OUTPUT_DIR/log_extract_raft_model$i.log"
    LOG_SCORE="$OUTPUT_DIR/log_score_raft_model$i.log"

    # 1. Inference
    if [ ! -f "$ANSWER_FILE" ]; then
        echo "Running inference for epoch $i..."
        CUDA_VISIBLE_DEVICES=2 python evaluation/infer_mathverse.py \
            --model-path "$MODEL_PATH" \
            --image-folder "$IMAGE_FOLDER" \
            --question-file "$QUESTION_FILE" \
            --answers-file "$ANSWER_FILE" \
            > "$LOG_INFER" 2>&1

        if [ $? -ne 0 ]; then
            echo "❌ Inference failed at epoch $i. Check $LOG_INFER"
            continue
        fi
    else
        echo "✅ Inference already completed for epoch $i"
    fi
    echo "Epoch $i completed!"
done