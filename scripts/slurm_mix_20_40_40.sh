#!/bin/bash
#SBATCH --job-name=mix-20-40-40
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=b200-node-2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=output/mix_20_40_40/slurm_%j.out
#SBATCH --error=output/mix_20_40_40/slurm_%j.err

source /NHNHOME/WORKSPACE/0426030085_A/boseong/env-boseong.sh
USERNAME=boseong
if [ -z "$USERNAME" ]; then echo "ERROR: \$USERNAME is not set." >&2; exit 1; fi

cd /NHNHOME/WORKSPACE/0426030085_A/boseong/Qwen-VL-Series-Finetune
source .venv/bin/activate

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_QWENVL_VIDEO_READER=decord

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME="/NHNHOME/WORKSPACE/0426030085_A/boseong/models/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="output/mix_20_40_40"

BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-1000}
EVAL_STEPS=${EVAL_STEPS:-1000}

# ── Data mixture: 20% MCQ / 40% BBox / 40% Counting  (total train = 120,000) ─
#
#   Dataset     | Available (train) | Target  | Ratio
#   ------------|-------------------|---------|-------
#   MCQ         |  98,204           |  24,000 |  24%
#   BBox        |  15,836           |  48,000 | 303%  (upsampled 3.0x)
#   Counting    | 526,470           |  48,000 |   9%
#   ------------|-------------------|---------|-------
#   TOTAL       |                   | 120,000 |
#
DATA_BASE="/NHNHOME/WORKSPACE/0426030085_A/dataset"
DATA_PATH="\
${DATA_BASE}/mcq/qwen/annotations_train.jsonl%24,\
${DATA_BASE}/refcoco/qwen/annotations_train.jsonl%303,\
${DATA_BASE}/counting/qwen/annotations_train.jsonl%9"

# Eval only on bbox and counting (IoU + exact_match)
EVAL_PATH="\
${DATA_BASE}/refcoco/qwen/annotations_val.jsonl%303,\
${DATA_BASE}/counting/qwen/annotations_val.jsonl%9"

mkdir -p $OUTPUT_DIR

MASTER_PORT=$(( RANDOM % 10000 + 20000 ))
deepspeed --master_port=$MASTER_PORT src/train/train_sft.py \
    --use_liger_kernel False \
    --lora_enable False \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --image_min_pixels $((256 * 32 * 32)) \
    --image_max_pixels $((256 * 32 * 32)) \
    --learning_rate $LR \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --logging_dir $OUTPUT_DIR/tb_logs \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --eval_strategy "steps" \
    --eval_steps $EVAL_STEPS \
    --per_device_eval_batch_size 4 \
    --generation_max_new_tokens 256
