"""
Simple VQA inference script for Qwen3-VL-8B (base or LoRA-merged).

Usage:
    # Base model
    python scripts/inference_vqa.py --image path/to/image.jpg --question "What is in this image?"

    # LoRA model
    python scripts/inference_vqa.py --image path/to/image.jpg --question "What is in this image?" \
        --model_path output/vqa_lora --model_base /home/ubuntu/models/Qwen3-VL-8B-Instruct

    # Video
    python scripts/inference_vqa.py --video path/to/video.mp4 --question "What happens in this video?"
"""
import sys
sys.path.insert(0, "src")

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils import load_pretrained_model, is_lora_model


def main():
    parser = argparse.ArgumentParser(description="VQA inference with Qwen3-VL")
    parser.add_argument("--model_path", type=str,
                        default="/home/ubuntu/models/Qwen3-VL-8B-Instruct",
                        help="Path to model or LoRA checkpoint")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Base model path (required for LoRA checkpoints)")
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--video", type=str, default=None, help="Path to video")
    parser.add_argument("--question", type=str, default="Describe this image in detail.",
                        help="Question to ask about the image/video")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.image is None and args.video is None:
        parser.error("Provide --image or --video")

    # Load model
    if is_lora_model(args.model_path) and args.model_base:
        print(f"Loading LoRA model from {args.model_path} (base: {args.model_base})")
        processor, model = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name="vqa_lora",
            use_flash_attn=True,
        )
    else:
        print(f"Loading base model from {args.model_path}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(args.model_path)

    # Build message
    content = []
    if args.image:
        content.append({"type": "image", "image": Image.open(args.image)})
    if args.video:
        content.append({"type": "video", "video": args.video, "fps": 1.0})
    content.append({"type": "text", "text": args.question})

    messages = [{"role": "user", "content": content}]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    # Generate
    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    response = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    print(f"\nQuestion: {args.question}")
    print(f"Answer:   {response}")


if __name__ == "__main__":
    main()
