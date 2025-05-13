#!/bin/bash

# FLEURS Evaluation Script
EVAL_SCRIPT="whisper_evaluation.py"
BATCH_SIZE=4

# Same models, different dataset
declare -A BASE_MODELS
BASE_MODELS["kb-whisper-small"]="KBLab/kb-whisper-small"
BASE_MODELS["kb-whisper-medium"]="KBLab/kb-whisper-medium"
BASE_MODELS["kb-whisper-large_elevenlabs"]="KBLab/kb-whisper-large"

for MODEL_NAME in "${!BASE_MODELS[@]}"; do
    BASE_MODEL="${BASE_MODELS[$MODEL_NAME]}"
    SAVE_NAME="fleurs_${MODEL_NAME}"

    echo "Evaluating $MODEL_NAME on FLEURS (save: $SAVE_NAME)"
    python "$EVAL_SCRIPT" "$MODEL_NAME" "$BASE_MODEL" "$SAVE_NAME" "$BATCH_SIZE"
done
