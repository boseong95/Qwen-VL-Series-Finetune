#!/bin/bash
#SBATCH --job-name=suturing-mix-det
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=b200-node-3
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=output/suturing_mix_det/slurm_%j.out
#SBATCH --error=output/suturing_mix_det/slurm_%j.err

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
OUTPUT_DIR="output/suturing_mix_det"

BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}

# ── Data mixture: 50% Suturing / 50% BBox detection (RefCOCO) ─────────────────
#
#   Dataset     | Available  | Ratio
#   ------------|------------|-------
#   Suturing    | ~N         | 100%   (all suturing samples)
#   BBox (RC)   |  15,836    | matched to ~suturing count via ratio
#
# Ratio 379 ≈ "use ~60k from RefCOCO (3.79× its 15836 samples)"
# Adjust SUTURING_BASE to where the dataset lives on the server.
#
SUTURING_BASE="/NHNHOME/WORKSPACE/0426030085_A/dataset/suturing"
DATA_BASE="/NHNHOME/WORKSPACE/0426030085_A/dataset"

DATA_PATH="\
${SUTURING_BASE}/vqa_dataset/train.json%100,\
${DATA_BASE}/refcoco/qwen/annotations_train.jsonl%379"

EVAL_PATH="\
${SUTURING_BASE}/vqa_dataset/val.json%100,\
${DATA_BASE}/refcoco/qwen/annotations_val.jsonl%379"

IMAGE_FOLDER="${SUTURING_BASE}"

mkdir -p $OUTPUT_DIR

MASTER_PORT=$(( RANDOM % 10000 + 20000 ))
deepspeed --master_port=$MASTER_PORT src/train/train_sft.py \
    --use_liger_kernel False \
    --lora_enable False \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --image_folder $IMAGE_FOLDER \
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
    --image_max_pixels $((512 * 32 * 32)) \
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
