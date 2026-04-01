#!/bin/bash
set -e

MODEL_NAME="/home/ubuntu/models/Qwen3-VL-8B-Instruct"
BATCH_SIZE=${1:-1}

export PYTHONPATH=src:$PYTHONPATH

# Single GPU training with DeepSpeed ZeRO-2
deepspeed --num_gpus=1 src/train/train_sft.py \
    --use_liger_kernel False \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path test_data/train.json \
    --image_folder test_data/images \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/test_lora_bs${BATCH_SIZE} \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --image_min_pixels $((256 * 32 * 32)) \
    --image_max_pixels $((512 * 32 * 32)) \
    --learning_rate 1e-4 \
    --merger_lr 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to none \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --dataloader_num_workers 1 \
    --max_seq_length 512
