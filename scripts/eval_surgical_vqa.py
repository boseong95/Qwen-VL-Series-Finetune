"""
Run inference on the surgical VQA validation set and evaluate.

Usage:
  PYTHONPATH=src:$PYTHONPATH python scripts/eval_surgical_vqa.py \
    --checkpoint output/surgical_vqa_lora/checkpoint-5500 \
    --val_path surgical_vqa_data/vrtb_suturing_val.json \
    --viz_dir surgical_vqa_data/val_visualizations_json_example_prediction
"""

import argparse, json, os, re, sys, textwrap
import torch
import av
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset.data_utils import get_video_info, _extract_frames_pyav


def load_model(model_id, checkpoint_path):
    from peft import PeftModel
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    from transformers import AutoProcessor

    print(f"Loading base model: {model_id}")
    model = ModelClass.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    print(f"Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_inference(model, processor, sample, fps=1, max_frames=16,
                  video_min_pixels=65536, video_max_pixels=131072, image_patch_size=14):
    """Run inference by extracting frames with PyAV and passing as images.

    Workaround: Qwen3-VL generate() has a bug with video inputs where the
    chat template creates timestamp-separated video blocks but video_grid_thw
    only has one entry, causing StopIteration in RoPE position computation.
    Passing frames as multiple images avoids this entirely.
    """
    video_path = sample.get("video", "")
    vs = sample.get("video_start")
    ve = sample.get("video_end")
    question = sample["conversations"][0]["value"]

    # Extract frames with PyAV (same codec handling as training)
    frames = _extract_frames_pyav(video_path, fps, vs, ve)
    if len(frames) > max_frames:
        indices = [int(i * len(frames) / max_frames) for i in range(max_frames)]
        frames = [frames[i] for i in indices]

    # Build message: frames as images, question as text
    q_text = question.replace("<video>\n", "").replace("<image>\n", "")
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": q_text})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# ── Visualization helpers ─────────────────────────────────────────────────────
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13)
    FONT_TITLE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
except OSError:
    FONT = FONT_BOLD = FONT_TITLE = FONT_SMALL = ImageFont.load_default()

C_BG = (20, 20, 25); C_PANEL = (35, 35, 42); C_BORDER = (60, 60, 75)
C_CORRECT = (50, 180, 80); C_WRONG = (220, 60, 60)
C_JSON_KEY = (100, 200, 255); C_JSON_STR = (130, 255, 130)
C_JSON_BOOL = (255, 220, 80); C_JSON_NUM = (255, 180, 100); C_JSON_BRACE = (140, 140, 160)


def draw_rounded_rect(draw, xy, fill, radius=6):
    x0, y0, x1, y1 = xy
    draw.rectangle([x0+radius, y0, x1-radius, y1], fill=fill)
    draw.rectangle([x0, y0+radius, x1, y1-radius], fill=fill)
    for cx, cy, sa, ea in [(x0,y0,180,270),(x1-2*radius,y0,270,360),(x0,y1-2*radius,90,180),(x1-2*radius,y1-2*radius,0,90)]:
        draw.pieslice([cx, cy, cx+2*radius, cy+2*radius], sa, ea, fill=fill)


def draw_json_colored(draw, x, y, json_str, font, correct_keys=None, max_lines=14):
    try:
        pretty = json.dumps(json.loads(json_str), indent=2, ensure_ascii=False)
    except Exception:
        pretty = json_str
    for jline in pretty.split("\n")[:max_lines]:
        cx = x
        kv = re.match(r'^(\s*)"(.+?)":\s*(.+?),?\s*$', jline)
        if kv:
            indent_str, key, val = kv.groups()
            val_clean = val.rstrip(",")
            # Correctness indicator
            if correct_keys and key in correct_keys:
                ind = " \u2713" if correct_keys[key] else " \u2717"
                ind_c = C_CORRECT if correct_keys[key] else C_WRONG
            else:
                ind, ind_c = "", (100,100,100)
            draw.text((cx, y), indent_str+'"', fill=C_JSON_BRACE, font=font)
            cx += draw.textlength(indent_str+'"', font=font)
            draw.text((cx, y), key, fill=C_JSON_KEY, font=font)
            cx += draw.textlength(key, font=font)
            draw.text((cx, y), '":', fill=C_JSON_BRACE, font=font)
            cx += draw.textlength('":', font=font)
            draw.text((cx, y), ind, fill=ind_c, font=FONT_BOLD)
            cx += draw.textlength(ind, font=FONT_BOLD) + 2
            vc = C_JSON_STR if val_clean.startswith('"') else C_JSON_BOOL if val_clean in ("true","false","null") else C_JSON_NUM
            draw.text((cx, y), " "+val_clean, fill=vc, font=font)
        else:
            draw.text((cx, y), jline, fill=C_JSON_BRACE, font=font)
        y += 16
    return y


def make_pred_vs_gt_card(frame_img, sid, question, gt, pred, fmt, correct_keys=None):
    fw, fh = frame_img.size
    panel_w = 480
    card_w = fw + panel_w * 2 + 10
    card_h = max(fh, 560)
    card = Image.new("RGB", (card_w, card_h), C_BG)
    card.paste(frame_img, (0, (card_h - fh) // 2))
    draw = ImageDraw.Draw(card)

    # Q overlay on image
    q_clean = question.replace("<video>\n","")
    if "Choose from:" in q_clean: q_clean = q_clean.split("Choose from:")[0].strip() + "\n[18 phase choices...]"
    if "Choices:" in q_clean: q_clean = q_clean.split("Choices:")[0].strip() + "\n[18 phase choices...]"
    for tag in ["Respond with JSON:", "Respond in"]:
        if tag in question:
            q_clean += f"\n{tag} {question.split(tag)[-1].strip().split(chr(10))[0]}"
            break
    draw.rectangle([(0,0),(fw,55)], fill=(0,0,0))
    draw.text((5,3), f"[{fmt.upper()}] {sid}", fill=(100,180,255), font=FONT_SMALL)
    for qi, wl in enumerate(textwrap.wrap(q_clean.split("\n")[0], width=72)[:2]):
        draw.text((5, 17+qi*14), wl, fill=(210,210,220), font=FONT_SMALL)

    def _draw_answer_panel(x_start, y_start, title, title_color, answer_text, bg_color, ckeys=None):
        draw_rounded_rect(draw, (x_start, 5, x_start+panel_w-5, card_h-5), bg_color, radius=8)
        x, y = x_start+10, y_start
        draw.text((x, y), title, fill=title_color, font=FONT_TITLE); y += 24
        jm = re.search(r'(\{.*\}|\[.*\])', answer_text, re.DOTALL)
        cot = answer_text[:jm.start()].strip() if jm else ""
        if cot:
            for wl in textwrap.wrap(cot, width=55)[:2]:
                draw.text((x, y), wl, fill=(180,180,150), font=FONT_SMALL); y += 14
            y += 4
        if jm:
            y = draw_json_colored(draw, x, y, jm.group(0), FONT, correct_keys=ckeys, max_lines=16)
        else:
            for wl in textwrap.wrap(answer_text, width=55)[:10]:
                draw.text((x, y), wl, fill=(220,220,220), font=FONT); y += 16

    _draw_answer_panel(fw+5, 12, "GROUND TRUTH", (100,255,150), gt, C_PANEL)
    _draw_answer_panel(fw+panel_w+5, 12, "PREDICTION", (255,180,100), pred, (42,35,35), correct_keys)

    if correct_keys:
        nc = sum(correct_keys.values()); nt = len(correct_keys)
        bc = C_CORRECT if nc==nt else C_WRONG if nc==0 else (200,180,50)
        draw.text((fw+panel_w+15, card_h-25), f"{nc}/{nt} keys correct", fill=bc, font=FONT_BOLD)

    return card


def extract_mid_frame(video_path, vs, ve):
    mid = (vs + ve) / 2
    container = av.open(video_path)
    stream = container.streams.video[0]
    container.seek(int(mid * av.time_base), any_frame=False)
    for frame in container.decode(video=0):
        t = float(frame.pts * stream.time_base) if frame.pts is not None else 0
        if t >= mid - 0.5:
            img = frame.to_image()
            container.close()
            return img
    container.close()
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_id", default="/home/ubuntu/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--val_path", default="surgical_vqa_data/vrtb_suturing_val.json")
    parser.add_argument("--viz_dir", default=None)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.checkpoint, "eval_results.json")
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.checkpoint, "val_viz")
    os.makedirs(args.viz_dir, exist_ok=True)

    model, processor = load_model(args.model_id, args.checkpoint)

    with open(args.val_path) as f:
        val_data = json.load(f)
    print(f"\nRunning inference on {len(val_data)} val samples...")

    predictions, references, sample_ids, answer_formats = [], [], [], []

    for i, s in enumerate(val_data):
        sid = s["id"]
        fmt = s.get("answer_format", "unknown")
        gt = s["conversations"][1]["value"]
        print(f"  [{i+1}/{len(val_data)}] {sid} ({fmt})...", end=" ", flush=True)

        if not os.path.exists(s.get("video", "")):
            pred = "[ERROR: video not found]"
            print("SKIP")
        else:
            try:
                pred = run_inference(model, processor, s, fps=args.fps,
                                     max_frames=args.max_frames)
                print(f"OK ({len(pred)} chars)")
            except Exception as e:
                import traceback
                pred = f"[ERROR: {type(e).__name__}: {e}]"
                print(f"ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()

        predictions.append(pred)
        references.append(gt)
        sample_ids.append(sid)
        answer_formats.append(fmt)

    # Compute metrics
    from train.train_utils import compute_surgical_vqa_metrics
    results = compute_surgical_vqa_metrics(predictions, references, sample_ids, answer_formats)
    results["raw"] = [
        {"id": sid, "format": fmt, "prediction": pred, "reference": ref}
        for sid, fmt, pred, ref in zip(sample_ids, answer_formats, predictions, references)
    ]

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {args.output}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"JSON parse rate: {results.get('json_parse_rate', 'N/A')}")
    for fmt in ["json", "nl", "single"]:
        acc = results.get(f"{fmt}_accuracy")
        cnt = results.get(f"{fmt}_count", 0)
        if acc is not None:
            print(f"{fmt:>6} accuracy: {acc:.0%} ({cnt} samples)")
    print(f"\nPer-key accuracy:")
    for k, v in results["per_key"].items():
        mae = f"  MAE={v['mae']:.1f}" if "mae" in v else ""
        print(f"  {k:<25} {v['correct']}/{v['total']} = {v['accuracy']:.0%}{mae}")
    print(f"\nPer-category accuracy:")
    for cat, v in results["per_category"].items():
        print(f"  {cat:<25} {v['correct']}/{v['total']} = {v['accuracy']:.0%}")

    # Visualize JSON predictions
    print(f"\nGenerating visualizations -> {args.viz_dir}/")
    for i, s in enumerate(val_data):
        sid = s["id"]
        if s.get("answer_format") != "json":
            continue
        video_path = s.get("video", "")
        if not os.path.exists(video_path):
            continue
        frame = extract_mid_frame(video_path, s.get("video_start", 0), s.get("video_end", 10))
        if not frame:
            continue
        correct_keys = None
        for ps in results["per_sample"]:
            if ps["id"] == sid:
                correct_keys = ps.get("correct_keys", {})
                break
        card = make_pred_vs_gt_card(
            frame, sid, s["conversations"][0]["value"],
            references[i], predictions[i], "json", correct_keys)
        card.save(os.path.join(args.viz_dir, f"{sid}_pred.png"))

    n_viz = sum(1 for s in val_data if s.get("answer_format") == "json")
    print(f"Done. {n_viz} prediction cards saved.")


if __name__ == "__main__":
    main()
