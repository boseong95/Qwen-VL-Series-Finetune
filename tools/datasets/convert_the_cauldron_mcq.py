"""Convert HuggingFaceM4/the_cauldron MCQ subsets to LLaVA JSONL format.

Output: /NHNHOME/WORKSPACE/0426030085_A/dataset/mcq/qwen/
    annotations_train.jsonl
    annotations_val.jsonl
    images/<subset>/<idx>.jpg
"""
import argparse
import glob
import json
import os
import random
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm

MCQ_SUBSETS = ["scienceqa", "aokvqa", "ai2d", "iconqa", "raven", "visual7w", "tqa"]

SRC  = "/NHNHOME/WORKSPACE/0426030085_A/dataset/the_cauldron"
DST  = "/NHNHOME/WORKSPACE/0426030085_A/dataset/mcq/qwen"
VAL_RATIO = 0.10
SEED = 42


def extract_image(img_dict, save_path: str) -> bool:
    try:
        import numpy as np
        if isinstance(img_dict, dict) and "bytes" in img_dict:
            img = Image.open(BytesIO(img_dict["bytes"])).convert("RGB")
        elif isinstance(img_dict, bytes):
            img = Image.open(BytesIO(img_dict)).convert("RGB")
        elif isinstance(img_dict, np.ndarray):
            img = Image.fromarray(img_dict).convert("RGB")
        else:
            img = img_dict.convert("RGB")
        img.save(save_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  [warn] image save failed: {e}")
        return False


def convert_subset(subset: str, img_root: str) -> list:
    files = sorted(glob.glob(f"{SRC}/{subset}/train-*.parquet"))
    records = []
    for f in files:
        df = pd.read_parquet(f)
        for idx, row in df.iterrows():
            raw_imgs = row["images"]
            imgs = list(raw_imgs) if hasattr(raw_imgs, "__iter__") and not isinstance(raw_imgs, dict) else [raw_imgs]
            raw_turns = row["texts"]
            turns = list(raw_turns) if hasattr(raw_turns, "__iter__") and not isinstance(raw_turns, dict) else [raw_turns]

            # Save images
            img_paths = []
            for i, img in enumerate(imgs):
                fname = f"{subset}_{len(records)}_{i}.jpg"
                fpath = os.path.join(img_root, fname)
                if not os.path.exists(fpath):
                    extract_image(img, fpath)
                img_paths.append(fpath)

            # Build LLaVA conversations (turns may be numpy structured records)
            conversations = []
            for ti, turn in enumerate(turns):
                user_val = str(turn["user"])
                if ti == 0 and img_paths:
                    img_tags = "".join("<image>\n" for _ in img_paths)
                    user_val = img_tags + user_val
                conversations.append({"from": "human", "value": user_val})
                conversations.append({"from": "gpt",   "value": str(turn["assistant"])})

            record = {"conversations": conversations, "source": subset}
            if img_paths:
                record["image"] = img_paths if len(img_paths) > 1 else img_paths[0]
            records.append(record)
    return records


def main():
    os.makedirs(DST, exist_ok=True)
    img_root = os.path.join(DST, "images")
    os.makedirs(img_root, exist_ok=True)

    all_records = []
    for subset in MCQ_SUBSETS:
        print(f"Processing {subset} ...")
        recs = convert_subset(subset, img_root)
        print(f"  → {len(recs):,} samples")
        all_records.extend(recs)

    random.seed(SEED)
    random.shuffle(all_records)
    n_val = int(len(all_records) * VAL_RATIO)
    val_records   = all_records[:n_val]
    train_records = all_records[n_val:]

    for split, recs in [("train", train_records), ("val", val_records)]:
        path = os.path.join(DST, f"annotations_{split}.jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(recs):,} samples → {path}")


if __name__ == "__main__":
    main()
