#!/usr/bin/env python3
"""Standalone generation-based evaluation from a checkpoint.

Usage (2 GPUs via torchrun):
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 src/eval_ckpt.py \
        --model_id <checkpoint_path> \
        --eval_path <data_path> \
        --output_dir <output_dir> \
        ...
"""
import json
import os
import sys
from collections import Counter

import torch
from transformers import AutoProcessor, HfArgumentParser

from model.load_model import load_qwen_vl_generation_model
from dataset.sft_dataset import SupervisedDataset, DataCollatorForSupervisedDataset, _load_data_mixture
from trainer import QwenSFTTrainer
from params import DataArguments, ModelArguments, TrainingArguments


def compute_vqa_metrics(eval_pred) -> dict:
    predictions = eval_pred.predictions
    references = eval_pred.references

    def _token_f1(pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0
        p = num_common / len(pred_tokens)
        r = num_common / len(ref_tokens)
        return 2 * p * r / (p + r)

    def _extract_bbox(s: str):
        """Return bbox_2d list from JSON text, handling extra text and partial/truncated JSON."""
        import json as _j, re as _re
        try:
            return _j.loads(s).get("bbox_2d")
        except Exception:
            pass
        for m in _re.finditer(r'\{[^{}]*"bbox_2d"[^{}]*\}', s):
            try:
                return _j.loads(m.group()).get("bbox_2d")
            except Exception:
                pass
        m = _re.search(r'"bbox_2d"\s*:\s*\[([^\]]*)', s)
        if m:
            try:
                nums = [float(x.strip().rstrip('.')) for x in m.group(1).split(',') if x.strip() and x.strip().rstrip('.').replace('.','',1).lstrip('-').isdigit()]
                if len(nums) >= 4:
                    return [int(n) for n in nums[:4]]
            except Exception:
                pass
        return None

    def _bbox_iou(pred: str, ref: str) -> float:
        pb = _extract_bbox(pred)
        rb = _extract_bbox(ref)
        if not pb or not rb:
            return 0.0
        try:
            ix1 = max(pb[0], rb[0]); iy1 = max(pb[1], rb[1])
            ix2 = min(pb[2], rb[2]); iy2 = min(pb[3], rb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ap = max(0, pb[2] - pb[0]) * max(0, pb[3] - pb[1])
            ar = max(0, rb[2] - rb[0]) * max(0, rb[3] - rb[1])
            union = ap + ar - inter
            return inter / union if union > 0 else 0.0
        except Exception:
            return 0.0

    def _is_bbox(s: str) -> bool:
        return _extract_bbox(s) is not None

    bbox_iou, bbox_acc50, f1_scores, em_scores = [], [], [], []
    for p, r in zip(predictions, references):
        if _is_bbox(r):
            iou = _bbox_iou(p, r)
            bbox_iou.append(iou)
            bbox_acc50.append(float(iou >= 0.5))
        else:
            f1_scores.append(_token_f1(p, r))
            em_scores.append(float(p.strip().lower() == r.strip().lower()))

    metrics = {}
    if bbox_iou:
        n = len(bbox_iou)
        metrics["bbox_iou"] = sum(bbox_iou) / n
        metrics["bbox_acc50"] = sum(bbox_acc50) / n
    if f1_scores:
        n = len(f1_scores)
        metrics["token_f1"] = sum(f1_scores) / n
        metrics["exact_match"] = sum(em_scores) / n
    return metrics


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_path is None:
        raise ValueError("--eval_path is required for evaluation.")

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model = load_qwen_vl_generation_model(
        model_args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
    )
    model.config.use_cache = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    eval_dataset = SupervisedDataset(
        data_path=_load_data_mixture(data_args.eval_path),
        processor=processor,
        data_args=data_args,
        model_id=model_args.model_id,
    )
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_vqa_metrics,
    )

    # Derive step number from checkpoint directory name for the viz output path
    ckpt_name = os.path.basename(model_args.model_id.rstrip("/"))
    if ckpt_name.startswith("checkpoint-"):
        try:
            trainer.state.global_step = int(ckpt_name.split("-")[1])
        except (ValueError, IndexError):
            pass

    metrics = trainer.evaluate()

    local_rank = training_args.local_rank
    if local_rank in (-1, 0):
        eval_tag = training_args.eval_tag or f"step_{trainer.state.global_step}"
        metrics_dir = os.path.join(training_args.output_dir, "eval_viz", eval_tag)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n=== Eval Metrics ({eval_tag}) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\nSaved → {metrics_path}")


if __name__ == "__main__":
    main()
