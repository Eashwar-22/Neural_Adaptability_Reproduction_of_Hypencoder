#!/bin/bash

# Define base directory
BASE_DIR="outputs/inference"
cd $BASE_DIR || exit

echo "Organizing results in $BASE_DIR..."

# Function to move results
organize() {
    SUFFIX=$1
    TARGET_DIR=$2

    if ls *_${SUFFIX} 1> /dev/null 2>&1; then
        echo "Moving *_${SUFFIX} to ${TARGET_DIR}/..."
        mkdir -p "${TARGET_DIR}"
        mv *_${SUFFIX} "${TARGET_DIR}/"
    else
        echo "No files found for suffix: ${SUFFIX}"
    fi
}

# 1. Pretrained
organize "pretrained" "pretrained"

# 2. LoRA r8
organize "lora_r8" "lora_r8"

# 3. LoRA r64 (various ckpts)
organize "lora_r64" "lora_r64"
organize "lora_r64_v2" "lora_r64"
organize "lora_r64_alpha256" "lora_r64_alpha256"

# Handle specific cases if any (e.g. ckpts)
if ls *ckpt* 1> /dev/null 2>&1; then
   # Move remaining ckpt folders if they match known patterns, or leave them if complex
   echo "Checking remaining checkpoint folders..."
   mv *_ckpt* lora_r64/ 2>/dev/null
fi

echo "Organization complete."
ls -F
