#!/bin/bash
set -euo pipefail

# --- 1. 配置区域 (请修改这里) ---
INPUT_JSONL="/hpc2hdd/home/ycui785/dataset/llava-665k/llava_v1_5_mix665k_shuffled_fixed.jsonl"
IMAGE_ROOT="/hpc2hdd/home/ycui785/dataset/llava-665k/train_split"
MODEL_PATH="/hpc2hdd/home/ycui785/model/qwen2_5_vl_3b"

OUTPUT_NPY="feature_665/llava_final_embeddings.npy"
OUTPUT_JSONL="feature_665/llava_final_valid.jsonl"

# 并行配置
NUM_GPUS=1
BATCH_SIZE=2
CHUNK_SIZE=10000

# ✅ 新增：DataLoader / AMP 配置（建议）
NUM_WORKERS=1          # 每进程 workers；2卡一般 4~8 都行
PREFETCH_FACTOR=4      # 预取倍数（workers>0 才生效）
AMP_DTYPE="fp16"       # none|fp16|bf16（建议 fp16；A100/H100 可 bf16）

# 日志目录
LOG_DIR="logs_stage1_665"
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_NPY")"

echo "========================================================"
echo "🚀 Start Stage 1: Multimodal Feature Extraction"
echo "   Input:  $INPUT_JSONL"
echo "   GPUs:   $NUM_GPUS"
echo "   BS:     $BATCH_SIZE"
echo "   Chunk:  $CHUNK_SIZE"
echo "   Workers per GPU: $NUM_WORKERS"
echo "   Prefetch: $PREFETCH_FACTOR"
echo "   AMP: $AMP_DTYPE"
echo "========================================================"

# --- 2. 并行启动提取任务 (Extraction) ---
pids=()

for (( i=0; i<NUM_GPUS; i++ )); do
  echo "[Master] Launching Worker $i on GPU $i..."

  CUDA_VISIBLE_DEVICES=$i python -u stage1_transformers_checkpoint_dp.py \
    --input_jsonl "$INPUT_JSONL" \
    --image_root "$IMAGE_ROOT" \
    --output_npy "$OUTPUT_NPY" \
    --output_jsonl "$OUTPUT_JSONL" \
    --qwen_model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --chunk_size_samples "$CHUNK_SIZE" \
    --chunk_offset "$i" \
    --chunk_stride "$NUM_GPUS" \
    --encoder_type "qwen2vl" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --amp_dtype "$AMP_DTYPE" \
    > "$LOG_DIR/worker_${i}.log" 2>&1 &

  pids+=($!)
done

# --- 3. 等待所有任务完成 (Wait) ---
echo "--------------------------------------------------------"
echo "⏳ All $NUM_GPUS workers launched. Waiting for completion..."
echo "   Check logs in: $LOG_DIR/worker_*.log"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "❌ One or more workers failed. Please check logs in $LOG_DIR."
  exit 1
fi

echo "✅ All extraction workers finished!"

# --- 4. 执行合并 (Merge) ---
echo "--------------------------------------------------------"
echo "🔄 Starting Merge process..."

python -u stage1_transformers_checkpoint_dp.py \
  --input_jsonl "$INPUT_JSONL" \
  --output_npy "$OUTPUT_NPY" \
  --output_jsonl "$OUTPUT_JSONL" \
  --qwen_model_path "$MODEL_PATH" \
  --chunk_size_samples "$CHUNK_SIZE" \
  --merge_only

echo "========================================================"
echo "🎉 Stage 1 Complete!"
echo "   Embeddings saved to: $OUTPUT_NPY"
echo "========================================================"
