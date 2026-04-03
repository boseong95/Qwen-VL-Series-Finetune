#!/bin/bash
# ============================================================================
# Train Qwen3-VL-8B on image VQA with LoRA (single GPU)
#
# Dataset: LLaVA-Instruct-150K (156K train / 100 eval, COCO images)
# GPU:     ~33 GB VRAM at bs=1 (fits on 48GB+ GPU)
# ============================================================================

MODEL_NAME="/home/ubuntu/models/Qwen3-VL-8B-Instruct"
DATA_PATH="vqa_data/llava_instruct_train.json"
EVAL_PATH="vqa_data/llava_instruct_eval.json"
IMAGE_FOLDER="vqa_data/images"
OUTPUT_DIR="output/vqa_lora"

BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-1e-4}

export PYTHONPATH=src:$PYTHONPATH

# Qwen3-VL uses N * 32 * 32 for pixel calculation
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
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --image_min_pixels $((256 * 32 * 32)) \
    --image_max_pixels $((512 * 32 * 32)) \
    --learning_rate $LR \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
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
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --per_device_eval_batch_size 1 \
    --generation_max_new_tokens 256 \
    --metric_for_best_model "eval_token_f1" \
    --greater_is_better True \
    --load_best_model_at_end True
