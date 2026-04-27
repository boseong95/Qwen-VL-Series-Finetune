"""Quick sanity-check for the suturing VQA dataset.

Usage:
  cd /NHNHOME/WORKSPACE/0426030085_A/boseong/Qwen-VL-Series-Finetune
  source .venv/bin/activate
  python scripts/test_suturing_dataset.py --train_json /path/to/train.json \
      [--image_folder /path/to/images] [--n 20]
"""
import argparse
import json
import os
import sys

def check(train_json, image_folder, n):
    with open(train_json) as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")
    print(f"Keys in first record: {list(data[0].keys())}")
    print()

    missing_images = 0
    coord_samples = []
    ok = 0

    for i, item in enumerate(data[:n]):
        # Image check
        img = item.get("image")
        if isinstance(img, list):
            img = img[0]
        if img:
            path = img if os.path.exists(img) else os.path.join(image_folder or "", img)
            if not os.path.exists(path):
                missing_images += 1
                if missing_images <= 3:
                    print(f"  [MISSING IMAGE] {path}")

        # Conversation check
        convs = item.get("conversations", [])
        for turn in convs:
            if turn.get("from") == "gpt":
                val = turn.get("value", "")
                try:
                    parsed = json.loads(val)
                    if "bbox_2d" in parsed:
                        b = parsed["bbox_2d"]
                        coord_samples.append(b)
                        if max(b) > 1.0:
                            print(f"  [WARN] sample {i}: bbox not in [0,1]: {b}")
                except Exception:
                    pass
        ok += 1

    print(f"Checked {ok}/{min(n, len(data))} samples")
    print(f"Missing images: {missing_images}")
    if coord_samples:
        print(f"Bbox samples ({len(coord_samples)} found in first {n}):")
        for b in coord_samples[:5]:
            print(f"  {b}")
        vals = [v for b in coord_samples for v in b]
        print(f"  coord range: [{min(vals):.4f}, {max(vals):.4f}]")
    else:
        print("No bbox_2d answers found in sampled records — check answer format")

    print()
    print("Sample record:")
    print(json.dumps(data[0], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--image_folder", default=None)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    check(args.train_json, args.image_folder, args.n)
