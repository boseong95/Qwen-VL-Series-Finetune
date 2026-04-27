"""Convert allenai/pixmo-points parquet to LLaVA JSONL format.

Points are in percentage (0-100) → converted to 0-1000 (Qwen3-VL convention).
Images are URL-only: downloaded with a thread pool and cached to disk.

Output: /NHNHOME/WORKSPACE/0426030085_A/dataset/pixmo_points/qwen/
    annotations_train.jsonl
    annotations_val.jsonl
    images/<sha256[:2]>/<sha256>.jpg
"""
import glob
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

SRC  = "/NHNHOME/WORKSPACE/0426030085_A/dataset/pixmo-points/data"
DST  = "/NHNHOME/WORKSPACE/0426030085_A/dataset/pixmo_points/qwen"
VAL_RATIO  = 0.10
SEED       = 42
NUM_WORKERS = 32
TIMEOUT     = 10  # seconds per image request

QUESTION_TEMPLATES = {
    "counting": [
        "Point to all {label} in the image.",
        "Mark the location of each {label} you can see.",
        "Indicate all {label} present in this image.",
    ],
    "default": [
        "Point to {label} in the image.",
        "Where is {label}? Mark its location.",
        "Locate {label} in this image.",
    ],
}


def img_save_path(sha256: str) -> str:
    return os.path.join(DST, "images", sha256[:2], f"{sha256}.jpg")


def download_image(url: str, sha256: str) -> bool:
    """Download image from URL and save as JPEG. Returns True on success."""
    path = img_save_path(sha256)
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        resp = requests.get(url, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(path, "JPEG", quality=95)
        return True
    except Exception:
        return False


def convert_points(points) -> list:
    """Convert [{x, y},...] in 0-100 pct to [[x,y],...] in 0-1000."""
    result = []
    for p in points:
        x = round(float(p["x"]) * 10)
        y = round(float(p["y"]) * 10)
        result.append([
            max(0, min(1000, x)),
            max(0, min(1000, y)),
        ])
    return result


def build_record(row, rng: random.Random) -> dict | None:
    sha256 = str(row["image_sha256"])
    path   = img_save_path(sha256)
    if not os.path.exists(path):
        return None

    label  = str(row["label"])
    method = str(row.get("collection_method", "default"))
    templates = QUESTION_TEMPLATES.get(method, QUESTION_TEMPLATES["default"])
    question  = rng.choice(templates).format(label=label)

    pts = convert_points(row["points"])
    answer = json.dumps({"points": pts})

    return {
        "image": path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt",   "value": answer},
        ],
        "source": "pixmo_points",
    }


def main():
    os.makedirs(os.path.join(DST, "images"), exist_ok=True)

    files = sorted(glob.glob(f"{SRC}/train-*.parquet"))
    print(f"Loading {len(files)} parquet file(s)...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Total rows: {len(df):,}")

    # ── Phase 1: download images in parallel ─────────────────────────────────
    print(f"\nDownloading images with {NUM_WORKERS} workers...")
    rows = df.to_dict("records")
    download_tasks = [(r["image_url"], r["image_sha256"]) for r in rows]

    success = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(download_image, url, sha): i
                   for i, (url, sha) in enumerate(download_tasks)}
        with tqdm(total=len(futures), unit="img") as bar:
            for fut in as_completed(futures):
                if fut.result():
                    success += 1
                bar.update(1)
                bar.set_postfix(ok=success)

    print(f"Downloaded: {success:,} / {len(rows):,}")

    # ── Phase 2: build JSONL records ─────────────────────────────────────────
    print("\nBuilding JSONL records...")
    rng = random.Random(SEED)
    records = []
    skipped = 0
    for row in tqdm(rows):
        rec = build_record(row, rng)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    print(f"Records: {len(records):,}  skipped (no image): {skipped:,}")

    # ── Phase 3: train/val split ──────────────────────────────────────────────
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
        print(f"{split}: {len(recs):,} → {path}")


if __name__ == "__main__":
    main()
