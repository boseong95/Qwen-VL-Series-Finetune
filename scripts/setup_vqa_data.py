"""
Download and prepare a small VQA dataset for fine-tuning.

Image VQA: Subset of LLaVA-Instruct-150K (COCO images)
  - Community gold standard for VLM instruction tuning
  - https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K

Video VQA: Subset of ActivityNet-based QA from LLaVA-Video-178K
  - https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K

Usage:
    python setup_vqa_data.py --num_image_samples 1000 --num_video_samples 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_llava_instruct_150k(output_dir: str, num_samples: int):
    """Download LLaVA-Instruct-150K JSON and a subset of COCO images."""
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import urllib.request

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Download the instruction JSON
    print("Downloading LLaVA-Instruct-150K JSON...")
    json_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="llava_instruct_150k.json",
        repo_type="dataset",
        local_dir=output_dir,
    )

    with open(json_path, "r") as f:
        full_data = json.load(f)

    print(f"Full dataset: {len(full_data)} samples")

    # Take a subset
    subset = full_data[:num_samples]

    # Collect unique image filenames
    image_files = set()
    for item in subset:
        if "image" in item:
            image_files.add(item["image"])

    print(f"Need to download {len(image_files)} COCO images...")

    # Download COCO images
    coco_base_url = "http://images.cocodataset.org/train2014"
    failed = []

    def download_image(img_name):
        dest = os.path.join(images_dir, img_name)
        if os.path.exists(dest):
            return img_name, True
        # COCO URLs use "COCO_train2014_" prefix, but the dataset JSON has bare filenames
        if not img_name.startswith("COCO_"):
            url_name = f"COCO_train2014_{img_name}"
        else:
            url_name = img_name
        url = f"{coco_base_url}/{url_name}"
        try:
            urllib.request.urlretrieve(url, dest)
            return img_name, True
        except Exception as e:
            return img_name, False

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_image, img): img for img in image_files}
        done = 0
        for future in as_completed(futures):
            img_name, success = future.result()
            done += 1
            if not success:
                failed.append(img_name)
            if done % 50 == 0:
                print(f"  Downloaded {done}/{len(image_files)} images...")

    # Filter out samples with failed images
    if failed:
        print(f"Warning: {len(failed)} images failed to download, filtering them out.")
        failed_set = set(failed)
        subset = [item for item in subset if item.get("image") not in failed_set]

    # Save subset JSON in LLaVA format (already in correct format)
    subset_path = os.path.join(output_dir, "llava_instruct_subset.json")
    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"Saved {len(subset)} image VQA samples to {subset_path}")
    print(f"Images in {images_dir}")
    return subset_path, images_dir


def download_video_vqa_subset(output_dir: str, num_samples: int):
    """
    Download a small video VQA subset from LLaVA-Video-178K (ActivityNet QA).
    Videos are downloaded from ActivityNet URLs.
    """
    import urllib.request

    os.makedirs(output_dir, exist_ok=True)
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print("Downloading LLaVA-Video-178K metadata...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "lmms-lab/LLaVA-Video-178K",
            name="ActivityNet_QA_Captions_open_ended",
            split=f"train[:{num_samples}]",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Could not download video dataset: {e}")
        print("Skipping video VQA subset. You can add video data manually later.")
        return None, None

    # Convert to LLaVA format and download videos
    llava_data = []
    downloaded = 0

    for i, item in enumerate(ds):
        video_id = item.get("id", f"video_{i}")
        video_url = item.get("video", "")
        conversations = item.get("conversations", [])

        if not video_url or not conversations:
            continue

        # Download video
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(videos_dir, video_filename)

        if not os.path.exists(video_path):
            try:
                urllib.request.urlretrieve(video_url, video_path)
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"  Downloaded {downloaded} videos...")
            except Exception:
                continue

        # Convert conversations to LLaVA format
        llava_convs = []
        for conv in conversations:
            role = conv.get("from", "")
            value = conv.get("value", "")
            if role in ("human", "gpt"):
                llava_convs.append({"from": role, "value": value})

        if llava_convs:
            llava_data.append({
                "id": video_id,
                "video": video_filename,
                "conversations": llava_convs,
            })

    subset_path = os.path.join(output_dir, "video_vqa_subset.json")
    with open(subset_path, "w") as f:
        json.dump(llava_data, f, indent=2)

    print(f"Saved {len(llava_data)} video VQA samples to {subset_path}")
    print(f"Videos in {videos_dir}")
    return subset_path, videos_dir


def merge_datasets(image_json, video_json, output_path):
    """Merge image and video VQA datasets into one JSON."""
    data = []
    if image_json and os.path.exists(image_json):
        with open(image_json) as f:
            data.extend(json.load(f))
    if video_json and os.path.exists(video_json):
        with open(video_json) as f:
            data.extend(json.load(f))

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nMerged dataset: {len(data)} samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download VQA datasets for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="vqa_data",
                        help="Output directory for dataset")
    parser.add_argument("--num_image_samples", type=int, default=1000,
                        help="Number of image VQA samples to download (from LLaVA-Instruct-150K)")
    parser.add_argument("--num_video_samples", type=int, default=100,
                        help="Number of video VQA samples to download")
    parser.add_argument("--skip_video", action="store_true",
                        help="Skip video VQA download (videos are large)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Image VQA
    print("=" * 60)
    print("Downloading Image VQA (LLaVA-Instruct-150K subset)")
    print("=" * 60)
    image_json, images_dir = download_llava_instruct_150k(
        output_dir, args.num_image_samples
    )

    # Video VQA
    video_json = None
    if not args.skip_video:
        print("\n" + "=" * 60)
        print("Downloading Video VQA (LLaVA-Video-178K ActivityNet subset)")
        print("=" * 60)
        video_json, videos_dir = download_video_vqa_subset(
            output_dir, args.num_video_samples
        )

    # Merge
    merged_path = os.path.join(output_dir, "vqa_train.json")
    merge_datasets(image_json, video_json, merged_path)

    print("\n" + "=" * 60)
    print("DONE! Dataset ready for training.")
    print(f"  Image VQA JSON: {image_json}")
    if video_json:
        print(f"  Video VQA JSON: {video_json}")
    print(f"  Merged JSON:    {merged_path}")
    print(f"  Image folder:   {images_dir}")
    print("=" * 60)
    print()
    print("To train (image-only, LoRA):")
    print(f"  bash scripts/train_vqa_lora.sh")
    print()
    print("To train (mixed image+video, LoRA):")
    print(f"  bash scripts/train_vqa_mixed_lora.sh")


if __name__ == "__main__":
    main()
