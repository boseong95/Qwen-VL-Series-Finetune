#!/bin/bash
# ============================================================================
# Train Qwen3-VL-8B on mixed surgical VQA with LoRA (single GPU)
#
# Dataset: VRTB-Suturing + COCO + Cholec80-VQA (configurable mix)
# Supports video_start/video_end for temporal clips
# ============================================================================

MODEL_NAME="/home/ubuntu/models/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="output/surgical_vqa_lora"

# ── Dataset mixing (override via env vars) ────────────────────────────────────
VRTB_RATIO=${VRTB_RATIO:-0.50}
COCO_RATIO=${COCO_RATIO:-0.25}
CHOLEC_RATIO=${CHOLEC_RATIO:-0.25}
MIX_TOTAL=${MIX_TOTAL:-0}  # 0 = auto (use all data proportionally)

# ── Training hyperparams (override via env vars) ──────────────────────────────
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-1e-4}
MAX_STEPS=${MAX_STEPS:-0}  # 0 = use epochs
SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH="surgical_vqa_data/mixed_train.json"
EVAL_PATH="surgical_vqa_data/vrtb_suturing_val.json"
IMAGE_FOLDER="vqa_data/images"  # for COCO images

export PYTHONPATH=src:$PYTHONPATH
export FORCE_QWENVL_VIDEO_READER=decord

# ── Step 1: Mix datasets ─────────────────────────────────────────────────────
echo "=== Mixing datasets (VRTB=${VRTB_RATIO}, COCO=${COCO_RATIO}, Cholec80=${CHOLEC_RATIO}) ==="
MIX_ARGS="--vrtb ${VRTB_RATIO} --coco ${COCO_RATIO} --cholec80 ${CHOLEC_RATIO}"
if [ "$MIX_TOTAL" -gt 0 ]; then
    MIX_ARGS="${MIX_ARGS} --total ${MIX_TOTAL}"
fi
python3 surgical_vqa_data/mix_datasets.py ${MIX_ARGS} --output ${DATA_PATH}

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Failed to generate mixed dataset"
    exit 1
fi

echo "=== Dataset ready: $(python3 -c "import json; print(len(json.load(open('${DATA_PATH}'))))" 2>/dev/null) samples ==="

# ── Step 2: Train ─────────────────────────────────────────────────────────────
TRAIN_ARGS=""
if [ "$MAX_STEPS" -gt 0 ]; then
    TRAIN_ARGS="--max_steps ${MAX_STEPS}"
fi

deepspeed --num_gpus=1 src/train/train_sft.py \
    --use_liger_kernel False \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --image_folder $IMAGE_FOLDER \
    --eval_image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --image_min_pixels $((256 * 32 * 32)) \
    --image_max_pixels $((512 * 32 * 32)) \
    --video_min_pixels $((128 * 32 * 32)) \
    --video_max_pixels $((256 * 32 * 32)) \
    --fps 2 \
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
    --per_device_eval_batch_size 1 \
    --generation_max_new_tokens 256 \
    --metric_for_best_model "eval_token_f1" \
    --greater_is_better True \
    --load_best_model_at_end True \
    $TRAIN_ARGS
