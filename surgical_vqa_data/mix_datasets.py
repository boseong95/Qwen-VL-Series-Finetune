"""
Mix multiple VQA datasets into a single LLaVA-format training JSON.

Configurable ratios via command line:
  python mix_datasets.py --vrtb 0.50 --coco 0.25 --cholec80 0.25

Outputs:
  - surgical_vqa_data/mixed_train.json
"""

import json, os, random, argparse
from pathlib import Path

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def sample_dataset(data, target_count):
    """Sample target_count items from data (with replacement if needed)."""
    if len(data) == 0:
        return []
    if target_count <= len(data):
        return random.sample(data, target_count)
    # Oversample: full copies + remainder
    n_full = target_count // len(data)
    remainder = target_count % len(data)
    result = data * n_full + random.sample(data, remainder)
    random.shuffle(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Mix VQA datasets")
    parser.add_argument("--vrtb", type=float, default=0.50,
                        help="VRTB-Suturing fraction (default: 0.50)")
    parser.add_argument("--coco", type=float, default=0.25,
                        help="COCO LLaVA-Instruct fraction (default: 0.25)")
    parser.add_argument("--cholec80", type=float, default=0.25,
                        help="Cholec80-VQA fraction (default: 0.25)")
    parser.add_argument("--total", type=int, default=None,
                        help="Total samples (default: use smallest dataset to compute)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: surgical_vqa_data/mixed_train.json)")
    args = parser.parse_args()

    # Normalize ratios
    total_ratio = args.vrtb + args.coco + args.cholec80
    r_vrtb = args.vrtb / total_ratio
    r_coco = args.coco / total_ratio
    r_cholec = args.cholec80 / total_ratio

    print(f"Mix ratios: VRTB={r_vrtb:.0%}, COCO={r_coco:.0%}, Cholec80={r_cholec:.0%}")

    # Load datasets
    datasets = {}

    vrtb_path = os.path.join(BASE_DIR, "vrtb_suturing_train.json")
    if os.path.exists(vrtb_path) and r_vrtb > 0:
        print(f"Loading VRTB-Suturing: {vrtb_path}")
        datasets["vrtb"] = (load_json(vrtb_path), r_vrtb)
        print(f"  {len(datasets['vrtb'][0]):,} samples")
    elif r_vrtb > 0:
        print(f"WARNING: {vrtb_path} not found, run generate_vrtb_suturing_full.py first")

    coco_path = os.path.join(ROOT_DIR, "vqa_data", "llava_instruct_train.json")
    if os.path.exists(coco_path) and r_coco > 0:
        print(f"Loading COCO LLaVA-Instruct: {coco_path}")
        datasets["coco"] = (load_json(coco_path), r_coco)
        print(f"  {len(datasets['coco'][0]):,} samples")
    elif r_coco > 0:
        print(f"WARNING: {coco_path} not found")

    cholec_path = os.path.join(BASE_DIR, "cholec80_vqa_train.json")
    if os.path.exists(cholec_path) and r_cholec > 0:
        print(f"Loading Cholec80-VQA: {cholec_path}")
        datasets["cholec80"] = (load_json(cholec_path), r_cholec)
        print(f"  {len(datasets['cholec80'][0]):,} samples")
    elif r_cholec > 0:
        print(f"WARNING: {cholec_path} not found, run generate_cholec80_vqa.py first")

    if not datasets:
        print("ERROR: No datasets loaded")
        return

    # Compute target total
    if args.total:
        total = args.total
    else:
        # Use the dataset that would be most constraining
        # (i.e., the one with the smallest count/ratio)
        max_possible = []
        for name, (data, ratio) in datasets.items():
            if ratio > 0:
                max_possible.append(int(len(data) / ratio))
        total = min(max_possible)

    print(f"\nTarget total: {total:,}")

    # Sample each dataset
    mixed = []
    for name, (data, ratio) in datasets.items():
        n = int(total * ratio)
        sampled = sample_dataset(data, n)
        print(f"  {name}: {n:,} samples (from {len(data):,})")
        mixed.extend(sampled)

    random.shuffle(mixed)

    # Strip system messages (sft_dataset.py expects alternating human/gpt pairs)
    for s in mixed:
        convs = s.get("conversations", [])
        if convs and convs[0].get("from") == "system":
            s["conversations"] = convs[1:]

    # Resolve relative video paths to absolute
    for s in mixed:
        if "video" in s and not os.path.isabs(s["video"]):
            # Relative paths from the generator scripts are relative to BASE_DIR
            abs_path = os.path.normpath(os.path.join(BASE_DIR, s["video"]))
            if os.path.exists(abs_path):
                s["video"] = abs_path
            else:
                print(f"  WARNING: video not found: {abs_path}")

    print(f"\nFinal mixed dataset: {len(mixed):,} samples")

    # Save
    out_path = args.output or os.path.join(BASE_DIR, "mixed_train.json")
    with open(out_path, "w") as f:
        json.dump(mixed, f, ensure_ascii=False)
    print(f"Saved → {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
