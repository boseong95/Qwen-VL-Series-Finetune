import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from model.load_model import get_qwen_vl_generation_backbone, load_qwen_vl_generation_model
from trainer import QwenSFTTrainer
from dataset import make_supervised_data_module
from params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer, compute_vqa_metrics
import pathlib

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    backbone = get_qwen_vl_generation_backbone(model)
    vision_tower = backbone.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = backbone.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = backbone.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(backbone.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = backbone.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    backbone = get_qwen_vl_generation_backbone(model)
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = backbone.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    backbone = get_qwen_vl_generation_backbone(model)

    if k_llm and hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        for layer in backbone.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(backbone, "visual") and hasattr(backbone.visual, "blocks"):
        for blk in backbone.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = load_qwen_vl_generation_model(
        model_args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
        **bnb_model_from_pretrained_args,
    )
    if training_args.use_liger_kernel and model.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
        rank0_print(f"Disabling Liger kernel for unsupported model_type: {model.config.model_type}")
        training_args.use_liger_kernel = False
        if hasattr(training_args, "liger_kernel_config"):
            training_args.liger_kernel_config = None

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    # ── Eval-on-save callback ──────────────────────────────────────────────
    from transformers import TrainerCallback

    class EvalOnSaveCallback(TrainerCallback):
        """Run surgical VQA eval in-process after each checkpoint save."""
        def __init__(self, trainer_ref):
            self._trainer_ref = trainer_ref

        def on_save(self, args, state, control, **kwargs):
            if local_rank not in (0, -1):
                return
            val_path = getattr(data_args, "eval_path", None)
            if not val_path or not os.path.exists(val_path):
                return

            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            viz_dir = os.path.join(ckpt_dir, "val_viz")
            eval_out = os.path.join(ckpt_dir, "eval_results.json")
            os.makedirs(viz_dir, exist_ok=True)

            rank0_print(f"\n=== Running surgical VQA eval (step {state.global_step}) ===")
            try:
                import json as _json
                from dataset.data_utils import _extract_frames_pyav
                from train.train_utils import compute_surgical_vqa_metrics
                from qwen_vl_utils import process_vision_info
                import subprocess as _sp

                with open(val_path) as f:
                    val_data = _json.load(f)

                trainer_obj = self._trainer_ref()
                if trainer_obj is None:
                    rank0_print("  Eval skipped: trainer ref lost")
                    return
                unwrapped = trainer_obj.accelerator.unwrap_model(trainer_obj.model)

                unwrapped.eval()
                preds, refs, sids, fmts = [], [], [], []
                frames_cache = {}

                for idx, s in enumerate(val_data):
                    sid = s["id"]
                    fmt = s.get("answer_format", "unknown")
                    gt = s["conversations"][1]["value"]
                    video_path = s.get("video", "")

                    if not os.path.exists(video_path):
                        preds.append("[ERROR: video not found]")
                        refs.append(gt); sids.append(sid); fmts.append(fmt)
                        continue

                    try:
                        # Extract frames (same as training)
                        frames = _extract_frames_pyav(
                            video_path, target_fps=1,
                            video_start=s.get("video_start"),
                            video_end=s.get("video_end"),
                        )
                        if len(frames) > 8:
                            indices = [int(i * len(frames) / 8) for i in range(8)]
                            frames = [frames[i] for i in indices]

                        # Build as images (workaround for Qwen3-VL generate bug)
                        q_text = s["conversations"][0]["value"].replace("<video>\n", "").replace("<image>\n", "")
                        content = [{"type": "image", "image": f} for f in frames]
                        content.append({"type": "text", "text": q_text})
                        messages = [{"role": "user", "content": content}]

                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, _ = process_vision_info(messages)
                        inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
                        inputs = {k: v.to(unwrapped.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            out = unwrapped.generate(**inputs, max_new_tokens=512, do_sample=False)
                        input_len = inputs["input_ids"].shape[1]
                        pred = processor.batch_decode(out[:, input_len:], skip_special_tokens=True)[0].strip()
                        frames_cache[sid] = frames
                    except Exception as e:
                        pred = f"[ERROR: {e}]"

                    preds.append(pred); refs.append(gt); sids.append(sid); fmts.append(fmt)

                # Compute metrics
                results = compute_surgical_vqa_metrics(preds, refs, sids, fmts)
                results["raw"] = [
                    {"id": sid, "format": fmt, "prediction": p, "reference": r}
                    for sid, fmt, p, r in zip(sids, fmts, preds, refs)
                ]
                with open(eval_out, "w") as f:
                    _json.dump(results, f, indent=2, ensure_ascii=False)

                # Print summary + log to tensorboard
                tb_metrics = {}
                jp = results.get("json_parse_rate")
                if jp is not None:
                    tb_metrics["eval/json_parse_rate"] = jp
                rank0_print(f"  JSON parse: {jp}")
                for f in ["json", "nl", "single"]:
                    acc = results.get(f"{f}_accuracy")
                    if acc is not None:
                        tb_metrics[f"eval/{f}_accuracy"] = acc
                        rank0_print(f"  {f:>6} acc: {acc:.0%}")
                for cat, v in results.get("per_category", {}).items():
                    if v.get("accuracy") is not None:
                        tb_metrics[f"eval/{cat}"] = v["accuracy"]
                    rank0_print(f"  {cat}: {v['correct']}/{v['total']} = {v['accuracy']:.0%}")
                for key, v in results.get("per_key", {}).items():
                    if v.get("accuracy") is not None:
                        tb_metrics[f"eval_key/{key}"] = v["accuracy"]

                # Log to tensorboard via trainer
                trainer_obj = self._trainer_ref()
                if trainer_obj is not None:
                    trainer_obj.log(tb_metrics)

                # Save viz for JSON samples
                n_viz = 0
                for idx, s in enumerate(val_data):
                    sid = s["id"]
                    if s.get("answer_format") != "json" or sid not in frames_cache:
                        continue
                    mid = frames_cache[sid][len(frames_cache[sid])//2]
                    # Save input frames as video
                    ftmp = os.path.join(viz_dir, f"_tmp_{sid}")
                    os.makedirs(ftmp, exist_ok=True)
                    for fi, fr in enumerate(frames_cache[sid]):
                        fr.save(os.path.join(ftmp, f"{fi:04d}.png"))
                    _sp.run(["ffmpeg", "-y", "-framerate", "1", "-i", os.path.join(ftmp, "%04d.png"),
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
                             "-loglevel", "error", os.path.join(viz_dir, f"{sid}_input.mp4")], check=True)
                    for ff in os.listdir(ftmp): os.remove(os.path.join(ftmp, ff))
                    os.rmdir(ftmp)
                    # Save mid-frame for quick inspection
                    mid.save(os.path.join(viz_dir, f"{sid}_frame.jpg"))
                    n_viz += 1

                rank0_print(f"  Saved {n_viz} viz to {viz_dir}")
                rank0_print(f"  Results -> {eval_out}")

            except Exception as e:
                import traceback
                rank0_print(f"  Eval error: {e}")
                traceback.print_exc()

            # Put model back to train mode
            if unwrapped is not None:
                unwrapped.train()

    import weakref
    # Create trainer first, then attach callback with a weak ref to avoid circular ref
    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )
    trainer.add_callback(EvalOnSaveCallback(weakref.ref(trainer)))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
