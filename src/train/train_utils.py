from __future__ import annotations

import re
import transformers
import torch
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.sft_trainer import GenerativeEvalPrediction


def _normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.strip().lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_vqa_metrics(eval_pred: GenerativeEvalPrediction, rank0_print_fn=print) -> dict:
    """Compute VQA metrics: exact match, token-level F1, ROUGE-L, BLEU-4."""
    predictions = eval_pred.predictions
    references = eval_pred.references

    # --- Exact Match ---
    em_scores = []
    for pred, ref in zip(predictions, references):
        em_scores.append(float(_normalize_answer(pred) == _normalize_answer(ref)))

    # --- Token-level F1 ---
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _normalize_answer(pred).split()
        ref_tokens = _normalize_answer(ref).split()
        common = set(pred_tokens) & set(ref_tokens)
        if len(common) == 0:
            f1_scores.append(0.0)
        else:
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            f1_scores.append(2 * precision * recall / (precision + recall))

    # --- ROUGE-L ---
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
    except ImportError:
        avg_rouge = 0.0

    # --- BLEU-4 ---
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
            bleu_scores.append(score)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
    except ImportError:
        avg_bleu = 0.0

    n = len(predictions)
    # Log a few samples for qualitative inspection
    rank0_print_fn("\n--- Eval Samples ---")
    for i in range(min(3, n)):
        rank0_print_fn(f"  Pred: {predictions[i][:120]}")
        rank0_print_fn(f"  Ref:  {references[i][:120]}")
        rank0_print_fn()

    return {
        "exact_match": sum(em_scores) / n,
        "token_f1": sum(f1_scores) / n,
        "rouge_l": avg_rouge,
        "bleu4": avg_bleu,
    }


# ── Surgical VQA Evaluation ──────────────────────────────────────────────────

def _extract_json(text: str):
    """Extract the first JSON object/array from a string, return parsed dict or None."""
    import json as _json
    # Try to find JSON in the text (after optional CoT prefix)
    for pattern in [r'\{[^{}]*\}', r'\{.*\}', r'\[.*\]']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return _json.loads(match.group(0))
            except _json.JSONDecodeError:
                continue
    # Try parsing the whole thing
    try:
        return _json.loads(text.strip())
    except _json.JSONDecodeError:
        return None


def compute_surgical_vqa_metrics(
    predictions: list[str],
    references: list[str],
    sample_ids: list[str] | None = None,
    answer_formats: list[str] | None = None,
) -> dict:
    """Compute per-key metrics for surgical VQA evaluation.

    Returns a dict with:
      - Overall: json_parse_rate, exact_match, token_f1
      - Per-key accuracy for JSON keys (phase, success, progress, etc.)
      - Per-category aggregate scores
      - Per-sample details for debugging

    Args:
        predictions: model output strings
        references: ground truth strings
        sample_ids: optional sample IDs for reporting
        answer_formats: optional per-sample format ("json", "nl", "single")
    """
    import json as _json
    n = len(predictions)
    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(n)]
    if answer_formats is None:
        answer_formats = ["unknown"] * n

    # ── Per-sample results ────────────────────────────────────────────────
    per_sample = []

    # ── Aggregate counters ────────────────────────────────────────────────
    # Per-key: {key: {"correct": int, "total": int, "errors": list}}
    key_metrics = {}
    # Per-format: {fmt: {"correct": int, "total": int}}
    format_metrics = {"json": {"correct": 0, "total": 0},
                      "nl": {"correct": 0, "total": 0},
                      "single": {"correct": 0, "total": 0}}
    # JSON parse tracking
    json_parse_ok = 0
    json_parse_total = 0

    # Category grouping for keys
    CATEGORY_MAP = {
        "phase": "phase_recognition",
        "current_phase": "phase_recognition",
        "phase_index": "phase_recognition",
        "failed_phase": "phase_recognition",
        "progress": "progress_estimation",
        "stage": "progress_estimation",
        "next_phase": "temporal_reasoning",
        "previous_phase": "temporal_reasoning",
        "is_transitioning": "temporal_reasoning",
        "success": "success_failure",
        "failure_reason": "success_failure",
        "was_retried": "retry_detection",
        "retry_successful": "retry_detection",
        "total_phases": "procedural_ordering",
        "phases_remaining": "procedural_ordering",
        "failures": "success_failure",
        "failed_phase": "success_failure",
    }

    def _key_match(pred_val, ref_val, key_name: str) -> bool:
        """Compare a predicted value to reference for a given key."""
        if ref_val is None:
            return pred_val is None
        if isinstance(ref_val, bool):
            if isinstance(pred_val, bool):
                return pred_val == ref_val
            # Handle string "true"/"false"
            if isinstance(pred_val, str):
                return pred_val.lower().strip() == str(ref_val).lower()
            return False
        if isinstance(ref_val, (int, float)):
            if isinstance(pred_val, (int, float)):
                # For progress: allow ±10 tolerance
                if key_name == "progress":
                    return abs(pred_val - ref_val) <= 10
                # For indices/counts: exact match
                return pred_val == ref_val
            return False
        if isinstance(ref_val, str):
            if not isinstance(pred_val, str):
                return False
            return _normalize_answer(pred_val) == _normalize_answer(ref_val)
        if isinstance(ref_val, list):
            if not isinstance(pred_val, list):
                return False
            # For failure lists: compare as sets of (phase, reason) tuples
            if ref_val and isinstance(ref_val[0], dict):
                ref_set = {(d.get("phase", ""), d.get("failure_reason", "")) for d in ref_val}
                pred_set = {(d.get("phase", ""), d.get("failure_reason", "")) for d in pred_val if isinstance(d, dict)}
                return ref_set == pred_set
            return pred_val == ref_val
        return str(pred_val) == str(ref_val)

    def _update_key(key, pred_val, ref_val, sid):
        if key not in key_metrics:
            key_metrics[key] = {"correct": 0, "total": 0, "mae_sum": 0.0, "mae_count": 0, "errors": []}
        km = key_metrics[key]
        km["total"] += 1
        correct = _key_match(pred_val, ref_val, key)
        if correct:
            km["correct"] += 1
        else:
            km["errors"].append({"id": sid, "pred": pred_val, "ref": ref_val})
        # MAE for numeric keys
        if isinstance(ref_val, (int, float)) and isinstance(pred_val, (int, float)):
            km["mae_sum"] += abs(pred_val - ref_val)
            km["mae_count"] += 1
        return correct

    # ── Evaluate each sample ──────────────────────────────────────────────
    for i in range(n):
        pred, ref, sid, fmt = predictions[i], references[i], sample_ids[i], answer_formats[i]
        sample_result = {"id": sid, "format": fmt, "correct_keys": {}, "parse_ok": True}

        if fmt == "json":
            json_parse_total += 1
            pred_json = _extract_json(pred)
            ref_json = _extract_json(ref)

            if pred_json is None:
                sample_result["parse_ok"] = False
                # Still count all ref keys as missed
                if ref_json and isinstance(ref_json, dict):
                    for k in ref_json:
                        _update_key(k, None, ref_json[k], sid)
                        sample_result["correct_keys"][k] = False
            else:
                json_parse_ok += 1
                # Compare each key
                if isinstance(ref_json, dict):
                    all_correct = True
                    for k, ref_v in ref_json.items():
                        pred_v = pred_json.get(k) if isinstance(pred_json, dict) else None
                        correct = _update_key(k, pred_v, ref_v, sid)
                        sample_result["correct_keys"][k] = correct
                        if not correct:
                            all_correct = False

            # Format-level tracking
            fmt_correct = sample_result["parse_ok"] and all(sample_result["correct_keys"].values())
            format_metrics["json"]["total"] += 1
            if fmt_correct:
                format_metrics["json"]["correct"] += 1

        elif fmt == "single":
            format_metrics["single"]["total"] += 1
            if _normalize_answer(pred) == _normalize_answer(ref):
                format_metrics["single"]["correct"] += 1
                sample_result["correct_keys"]["exact"] = True
            else:
                sample_result["correct_keys"]["exact"] = False

        elif fmt == "nl":
            format_metrics["nl"]["total"] += 1
            # Token F1 for NL
            pred_tokens = _normalize_answer(pred).split()
            ref_tokens = _normalize_answer(ref).split()
            common = set(pred_tokens) & set(ref_tokens)
            if common:
                prec = len(common) / len(pred_tokens) if pred_tokens else 0
                rec = len(common) / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            else:
                f1 = 0.0
            sample_result["token_f1"] = f1
            format_metrics["nl"]["correct"] += int(f1 > 0.5)

        per_sample.append(sample_result)

    # ── Aggregate results ─────────────────────────────────────────────────
    results = {
        "n_samples": n,
        "json_parse_rate": json_parse_ok / json_parse_total if json_parse_total > 0 else None,
    }

    # Per-format accuracy
    for fmt, fm in format_metrics.items():
        if fm["total"] > 0:
            results[f"{fmt}_accuracy"] = fm["correct"] / fm["total"]
            results[f"{fmt}_count"] = fm["total"]

    # Per-key accuracy
    results["per_key"] = {}
    for key, km in sorted(key_metrics.items()):
        entry = {
            "accuracy": km["correct"] / km["total"] if km["total"] > 0 else None,
            "correct": km["correct"],
            "total": km["total"],
        }
        if km["mae_count"] > 0:
            entry["mae"] = km["mae_sum"] / km["mae_count"]
        if km["errors"]:
            entry["errors"] = km["errors"][:3]  # first 3 errors for debugging
        results["per_key"][key] = entry

    # Per-category accuracy
    cat_scores = {}
    for key, km in key_metrics.items():
        cat = CATEGORY_MAP.get(key, "other")
        if cat not in cat_scores:
            cat_scores[cat] = {"correct": 0, "total": 0}
        cat_scores[cat]["correct"] += km["correct"]
        cat_scores[cat]["total"] += km["total"]

    results["per_category"] = {}
    for cat, cs in sorted(cat_scores.items()):
        results["per_category"][cat] = {
            "accuracy": cs["correct"] / cs["total"] if cs["total"] > 0 else None,
            "correct": cs["correct"],
            "total": cs["total"],
        }

    results["per_sample"] = per_sample

    return results


def maybe_zero_3(param, ignore_status=False, name=None, device=torch.device('cpu')):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if type(device) is str:
        device = torch.device(device)
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach()
    else:
        param = param.detach()
    if device == param.device:
        return param.clone()
    else:
        return param.to(device)

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)