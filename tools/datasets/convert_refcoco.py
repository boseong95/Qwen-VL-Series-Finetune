"""Convert lmms-lab/RefCOCO parquet to LLaVA JSONL format.

Coordinates are normalized to 0-1000 range (Qwen3-VL / Qwen2-VL convention).
RefCOCO bbox format: [x, y, w, h] absolute pixels → converted to [x1,y1,x2,y2] in 0-1000.

Output: /NHNHOME/WORKSPACE/0426030085_A/dataset/refcoco/qwen/
    annotations_train.jsonl
    annotations_val.jsonl
    images/<idx>.jpg
"""
import glob
import json
import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SRC  = "/NHNHOME/WORKSPACE/0426030085_A/dataset/RefCOCO"
DST  = "/NHNHOME/WORKSPACE/0426030085_A/dataset/refcoco/qwen"
VAL_RATIO = 0.10
SEED = 42

QUESTION_TEMPLATES = [
    "Locate {label} in the image and provide the bounding box in JSON format.",
    "Where is {label} in the image? Output the bounding box as JSON.",
    "Find {label} and give me its bounding box coordinates in JSON format.",
    "Detect {label} in this image. Return the result as JSON with a bbox_2d field.",
]


def bbox_xywh_to_xyxy_norm(bbox, img_w: int, img_h: int) -> list:
    """Convert [x,y,w,h] absolute px to [x1,y1,x2,y2] normalized 0-1000."""
    x, y, w, h = bbox
    x1 = round(x / img_w * 1000)
    y1 = round(y / img_h * 1000)
    x2 = round((x + w) / img_w * 1000)
    y2 = round((y + h) / img_h * 1000)
    return [
        max(0, min(1000, x1)),
        max(0, min(1000, y1)),
        max(0, min(1000, x2)),
        max(0, min(1000, y2)),
    ]


def extract_image(img_dict, save_path: str) -> tuple:
    """Save PIL image, return (width, height)."""
    try:
        if isinstance(img_dict, dict) and "bytes" in img_dict:
            img = Image.open(BytesIO(img_dict["bytes"])).convert("RGB")
        elif isinstance(img_dict, bytes):
            img = Image.open(BytesIO(img_dict)).convert("RGB")
        else:
            img = img_dict.convert("RGB")
        w, h = img.size
        img.save(save_path, "JPEG", quality=95)
        return w, h
    except Exception as e:
        print(f"  [warn] image save failed: {e}")
        return None, None


def main():
    os.makedirs(DST, exist_ok=True)
    img_root = os.path.join(DST, "images")
    os.makedirs(img_root, exist_ok=True)

    files = sorted(glob.glob(f"{SRC}/*.parquet") + glob.glob(f"{SRC}/**/*.parquet"))
    records = []
    rng = random.Random(SEED)

    for f in files:
        df = pd.read_parquet(f)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(f)):
            img_fname = f"{len(records)}.jpg"
            img_path  = os.path.join(img_root, img_fname)

            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img_w, img_h = img.size
                except Exception:
                    os.remove(img_path)
                    img_w, img_h = extract_image(row["image"], img_path)
            else:
                img_w, img_h = extract_image(row["image"], img_path)
            if img_w is None:
                continue

            # RefCOCO has multiple answer candidates — use first
            answers = row["answer"]
            label = answers[0] if hasattr(answers, "__len__") else str(answers)

            bbox_raw = row["bbox"]
            if hasattr(bbox_raw, "tolist"):
                bbox_raw = bbox_raw.tolist()
            bbox_norm = bbox_xywh_to_xyxy_norm(bbox_raw, img_w, img_h)

            question = rng.choice(QUESTION_TEMPLATES).format(label=label)
            answer   = json.dumps({"bbox_2d": bbox_norm})

            records.append({
                "image": img_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt",   "value": answer},
                ],
                "source": "refcoco",
            })

    random.seed(SEED)
    random.shuffle(records)
    n_val = int(len(records) * VAL_RATIO)
    val_records   = records[:n_val]
    train_records = records[n_val:]

    for split, recs in [("train", train_records), ("val", val_records)]:
        path = os.path.join(DST, f"annotations_{split}.jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(recs):,} samples → {path}")


if __name__ == "__main__":
    main()
