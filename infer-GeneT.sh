#!/bin/bash

# 默认参数设置
DEFAULT_CUDA_DEVICES="2,3"
DEFAULT_CKPT_DIR="GeneT-qwen0.5b-public-mode/"
DEFAULT_VAL_DATA="test.jsonl"
DEFAULT_MAX_MODEL_LEN=32768
DEFAULT_MAX_NEW_TOKENS=2
DEFAULT_TENSOR_PARALLEL=2

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  --cuda_devices=<值>      使用的GPU设备ID (默认: $DEFAULT_CUDA_DEVICES)"
    echo "  --ckpt_dir=<路径>        模型检查点目录 (默认: $DEFAULT_CKPT_DIR)"
    echo "  --val_data=<路径>        验证数据路径 (默认: $DEFAULT_VAL_DATA)"
    echo "  --max_model_len=<值>     最大模型长度 (默认: $DEFAULT_MAX_MODEL_LEN)"
    echo "  --max_new_tokens=<值>    最大生成token数 (默认: $DEFAULT_MAX_NEW_TOKENS)"
    echo "  --tensor_parallel=<值>   张量并行大小 (默认: $DEFAULT_TENSOR_PARALLEL)"
    echo "  --help                   显示此帮助信息"
    exit 0
}

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda_devices=*) CUDA_DEVICES="${1#*=}" ;;
        --ckpt_dir=*) CKPT_DIR="${1#*=}" ;;
        --val_data=*) VAL_DATA="${1#*=}" ;;
        --max_model_len=*) MAX_MODEL_LEN="${1#*=}" ;;
        --max_new_tokens=*) MAX_NEW_TOKENS="${1#*=}" ;;
        --tensor_parallel=*) TENSOR_PARALLEL="${1#*=}" ;;
        --help) show_help ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 设置参数值（使用默认值或用户提供的值）
CUDA_DEVICES=${CUDA_DEVICES:-$DEFAULT_CUDA_DEVICES}
CKPT_DIR=${CKPT_DIR:-$DEFAULT_CKPT_DIR}
VAL_DATA=${VAL_DATA:-$DEFAULT_VAL_DATA}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-$DEFAULT_MAX_NEW_TOKENS}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-$DEFAULT_TENSOR_PARALLEL}

# 执行推理命令
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
swift infer \
    --ckpt_dir "$CKPT_DIR" \
    --infer_backend vllm \
    --load_dataset_config false \
    --custom_val_dataset_path "$VAL_DATA" \
    --val_dataset_sample -1 \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --sft_type full \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --template_type qwen  # 根据模型类型固定为qwen

echo "推理完成!"
