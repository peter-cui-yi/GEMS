#!/bin/bash
set -euo pipefail

# --- 1. 配置区域 (请修改这里) ---
INPUT_JSONL="/hpc2hdd/home/ycui785/dataset/llava-665k/llava_v1_5_mix665k_shuffled_fixed.jsonl"
IMAGE_ROOT="/hpc2hdd/home/ycui785/dataset/llava-665k/train_split"

# ✅ 换成 Qwen3-VL-Embedding 模型（本地路径 or HF id）
MODEL_PATH="/hpc2hdd/home/ycui785/model/qwen3vl-emb"
# MODEL_PATH="/hpc2hdd/home/ycui785/model/Qwen3-VL-Embedding-2B"

OUTPUT_NPY="feature_665_qwen3/llava_final_embeddings.npy"
OUTPUT_JSONL="feature_665_qwen3/llava_final_valid.jsonl"

# 单卡配置
GPU_ID=0
BATCH_SIZE=4
CHUNK_SIZE=10000

# DataLoader / AMP
NUM_WORKERS=8
PREFETCH_FACTOR=4
AMP_DTYPE="fp16"     # none|fp16|bf16

# 是否启用 flash-attn2（你环境装了就开）
USE_FLASH_ATTN=0     # 0/1

# 日志目录
LOG_DIR="logs_stage1_665_qwen3"
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_NPY")"

echo "========================================================"
echo "🚀 Start Stage 1: Qwen3-VL-Embedding Feature Extraction (Single GPU)"
echo "   Input:  $INPUT_JSONL"
echo "   Root:   $IMAGE_ROOT"
echo "   Model:  $MODEL_PATH"
echo "   GPU:    $GPU_ID"
echo "   BS:     $BATCH_SIZE"
echo "   Chunk:  $CHUNK_SIZE"
echo "   Workers:$NUM_WORKERS"
echo "   Prefetch:$PREFETCH_FACTOR"
echo "   AMP:    $AMP_DTYPE"
echo "   FlashAttn:$USE_FLASH_ATTN"
echo "========================================================"

echo "--------------------------------------------------------"
echo "▶️  Running extraction on GPU $GPU_ID ..."
echo "   Log: $LOG_DIR/worker_${GPU_ID}.log"

CUDA_VISIBLE_DEVICES=$GPU_ID USE_FLASH_ATTN=$USE_FLASH_ATTN \
python -u stage1_transformers_checkpoint_dp.py \
  --input_jsonl "$INPUT_JSONL" \
  --image_root "$IMAGE_ROOT" \
  --output_npy "$OUTPUT_NPY" \
  --output_jsonl "$OUTPUT_JSONL" \
  --qwen_model_path "$MODEL_PATH" \
  --batch_size "$BATCH_SIZE" \
  --chunk_size_samples "$CHUNK_SIZE" \
  --chunk_offset 0 \
  --chunk_stride 1 \
  --encoder_type "qwen3vl_embed" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --amp_dtype "$AMP_DTYPE" \
  > "$LOG_DIR/worker_${GPU_ID}.log" 2>&1

echo "✅ Extraction finished!"

echo "--------------------------------------------------------"
echo "🔄 Starting Merge process..."
python -u stage1_transformers_checkpoint_dp.py \
  --input_jsonl "$INPUT_JSONL" \
  --output_npy "$OUTPUT_NPY" \
  --output_jsonl "$OUTPUT_JSONL" \
  --qwen_model_path "$MODEL_PATH" \
  --chunk_size_samples "$CHUNK_SIZE" \
  --merge_only \
  > "$LOG_DIR/merge.log" 2>&1

echo "========================================================"
echo "🎉 Stage 1 Complete!"
echo "   Embeddings saved to: $OUTPUT_NPY"
echo "   Valid jsonl saved to: $OUTPUT_JSONL"
echo "   Logs: $LOG_DIR"
echo "========================================================"
