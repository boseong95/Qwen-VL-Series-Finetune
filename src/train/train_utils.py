import re
import transformers
import torch
import logging

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