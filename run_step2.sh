#!/bin/bash

# ================= 配置区域 =================
# 输入数据路径 (请确保这是 Step 1 转换后的正确 jsonl 文件)
INPUT_JSONL="/hpc2hdd/home/ycui785/UnimaxMM/main_algo_code/features/llava_final_valid.jsonl" 

# 输出分数路径
OUTPUT_NPY="/hpc2hdd/home/ycui785/UnimaxMM/main_algo_code/features/llava_final_embeddings_scores.npy"

# 模型路径
MODEL_PATH="/hpc2hdd/home/ycui785/model/qwen2_5_vl_3b"

# 运行配置
NUM_GPUS=1           # 使用卡数
BATCH_SIZE=64        # 显存安全值，建议保持为 1

# 日志目录
LOG_DIR="logs_step2"
mkdir -p $LOG_DIR
mkdir -p $(dirname $OUTPUT_NPY)

# ===========================================

echo "========================================================"
echo "🚀 Start Step 2: Uncertainty Scoring (Robust Multi-GPU)"
echo "   Input:  $INPUT_JSONL"
echo "   Output: $OUTPUT_NPY"
echo "   Model:  $MODEL_PATH"
echo "   GPUs:   $NUM_GPUS"
echo "========================================================"

# 设置 PyTorch 显存优化参数，防止碎片化 OOM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 运行 Python 脚本
# 注意：这里不需要用 & 后台并行，因为你的 Python 脚本内部已经使用了 mp.Process 进行多进程管理
# 我们直接运行主进程即可，它会自己 spawn 出 8 个子进程

python step2_calc_uncertainty.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_npy "$OUTPUT_NPY" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --num_gpus $NUM_GPUS \
    2>&1 | tee "$LOG_DIR/step2_execution.log"

echo "========================================================"
if [ -f "$OUTPUT_NPY" ]; then
    echo "✅ Step 2 Completed Successfully!"
    echo "   Scores saved to: $OUTPUT_NPY"
else
    echo "❌ Step 2 Failed! Check logs at $LOG_DIR/step2_execution.log"
fi
echo "========================================================"