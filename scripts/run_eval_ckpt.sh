#!/bin/bash
# Evaluate a checkpoint on a fixed validation set.
#
# Usage:
#   bash scripts/run_eval_ckpt.sh <checkpoint_path> <eval_name> [gpu_ids]
#
# Examples:
#   bash scripts/run_eval_ckpt.sh output/mix_20_40_40/checkpoint-1329  ckpt1329
#   bash scripts/run_eval_ckpt.sh output/mix_50_25_25/checkpoint-1380  ckpt1380  4,5
#   bash scripts/run_eval_ckpt.sh output/mix_0_50_50                   final     6,7
#
# Outputs:
#   {output_dir}/eval_viz/{eval_name}/  ← jpg + json visualizations
#   {output_dir}/eval_viz/{eval_name}/metrics.json

set -e

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> <eval_name> [gpu_ids]}"
EVAL_NAME="${2:?Usage: $0 <checkpoint_path> <eval_name> [gpu_ids]}"
GPUS="${3:-4,5}"

cd /NHNHOME/WORKSPACE/0426030085_A/boseong/Qwen-VL-Series-Finetune
source .venv/bin/activate

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_QWENVL_VIDEO_READER=decord

# ── Fixed validation datasets (same for every checkpoint) ────────────────────
DATA_BASE="/NHNHOME/WORKSPACE/0426030085_A/dataset"
EVAL_PATH="\
${DATA_BASE}/refcoco/qwen/annotations_val.jsonl%100,\
${DATA_BASE}/counting/qwen/annotations_val.jsonl%9"

# ── Derive output dir from the checkpoint path ───────────────────────────────
CKPT_ABS=$(realpath "$CHECKPOINT")
if [[ "$(basename $CKPT_ABS)" == checkpoint-* ]]; then
    OUTPUT_DIR="$(dirname $CKPT_ABS)"
else
    OUTPUT_DIR="$CKPT_ABS"
fi

mkdir -p "$OUTPUT_DIR"
echo "Checkpoint : $CKPT_ABS"
echo "Eval name  : $EVAL_NAME"
echo "Output dir : $OUTPUT_DIR/eval_viz/$EVAL_NAME/"
echo "GPUs       : $GPUS"

MASTER_PORT=$(( RANDOM % 10000 + 20000 ))
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    src/eval_ckpt.py \
    --model_id "$CKPT_ABS" \
    --eval_path "$EVAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --eval_tag "$EVAL_NAME" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --lora_enable False \
    --per_device_eval_batch_size 4 \
    --generation_max_new_tokens 256 \
    --image_min_pixels $((256 * 32 * 32)) \
    --image_max_pixels $((256 * 32 * 32)) \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --report_to none
