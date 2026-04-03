"""
Convert Cholec80-VQA (and SurgicalGPT-Cholec80-VQA) text files into LLaVA JSON format.

Input:  {CHOLEC80_VQA}/Classification/{video_id}/{frame}_QA.txt  (pipe-delimited Q|A)
Output: cholec80_vqa_train.json  (LLaVA format, referencing videos with timestamps)

Since Cholec80-VQA and SurgicalGPT-Cholec80-VQA have identical content,
we use a single pass and deduplicate.

Frame numbers are at 25fps. We reference the source videos via video_start/video_end
(±0.5s window around the frame) to avoid extracting 21K images.
"""

import json, os, glob, re
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
CHOLEC80_VQA_DIR = "/home/ubuntu/datasets/vlm/Cholec80-VQA/Classification"
CHOLEC80_VIDEO_DIR = "/home/ubuntu/datasets/vlm/Cholec80/videos"
SOURCE_FPS = 25.0
FRAME_WINDOW_SEC = 0.5  # ±0.5s around the frame for video clip

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = (
    "You are a surgical video analysis assistant specialized in laparoscopic cholecystectomy procedures. "
    "When the user asks a question, respond in the exact format requested:\n"
    "- If the question specifies a JSON schema, first provide a brief reasoning, "
    "then output the JSON on a new line.\n"
    "- If the question asks for a single choice, respond with ONLY the chosen option.\n"
    "- If the question is open-ended, respond with a concise natural language description."
)

PHASE_NAMES = [
    "Preparation", "Calot Triangle Dissection", "Clipping & Cutting",
    "Gallbladder Dissection", "Gallbladder Packaging",
    "Cleaning & Coagulation", "Gallbladder Retraction",
]

TOOL_NAMES = [
    "Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag",
]

# ── Categorize questions ──────────────────────────────────────────────────────
def categorize_qa(question, answer):
    """Classify QA into types and generate format variants."""
    q_lower = question.lower().strip()
    a_lower = answer.lower().strip()

    variants = []

    if q_lower.startswith("what is the phase"):
        # Phase recognition → JSON + NL + single
        phase = answer.strip()
        variants.append(("json",
            f"{question}\n\nRespond with JSON: {{\"phase\": \"<name>\"}}",
            f"The surgical scene shows the {phase.lower()} stage.\n"
            f'{json.dumps({"phase": phase})}'))
        variants.append(("nl",
            f"{question}\n\nRespond in a complete sentence.",
            f"The current surgical phase is {phase}."))
        variants.append(("single", question, phase))

    elif q_lower.startswith("how many tools"):
        # Tool counting → JSON + single
        count = answer.strip()
        variants.append(("json",
            f"{question}\n\nRespond with JSON: {{\"tool_count\": <int>}}",
            f"Observing the surgical field, there are {count} tool(s) visible.\n"
            f'{json.dumps({"tool_count": int(count)})}'))
        variants.append(("single", question, count))

    elif q_lower.startswith("is ") and "used in" in q_lower:
        # Tool-phase presence → JSON + single
        yn = answer.strip()
        # Extract tool and phase from question
        match = re.match(r"is (\w[\w\s]*?) used in ([\w\s&]+)\?", q_lower)
        tool = match.group(1).strip() if match else "unknown"
        phase = match.group(2).strip() if match else "unknown"

        variants.append(("json",
            f"{question}\n\nRespond with JSON: {{\"tool\": \"<name>\", \"phase\": \"<name>\", \"present\": <bool>}}",
            f"Checking for {tool} during {phase}.\n"
            f'{json.dumps({"tool": tool, "phase": phase, "present": yn == "yes"})}'))
        variants.append(("single", question, yn))

    else:
        # Fallback: keep as single-answer
        variants.append(("single", question, answer.strip()))

    return variants


# ── Process all QA files ──────────────────────────────────────────────────────
samples = []
skipped = 0

video_dirs = sorted(glob.glob(os.path.join(CHOLEC80_VQA_DIR, "*")))
print(f"Found {len(video_dirs)} video directories")

for vdir in video_dirs:
    vid_num = os.path.basename(vdir)
    video_file = os.path.join(CHOLEC80_VIDEO_DIR, f"video{vid_num.zfill(2)}.mp4")

    if not os.path.exists(video_file):
        print(f"  WARNING: {video_file} not found, skipping video {vid_num}")
        skipped += 1
        continue

    video_rel = os.path.relpath(video_file, OUT_DIR)

    qa_files = sorted(glob.glob(os.path.join(vdir, "*_QA.txt")))
    for qa_file in qa_files:
        fname = os.path.basename(qa_file)
        frame_num = int(fname.split("_")[0])
        frame_sec = frame_num / SOURCE_FPS

        # Read QA pairs
        with open(qa_file) as f:
            lines = [l.strip() for l in f if l.strip() and "|" in l]

        for line_idx, line in enumerate(lines):
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            question, answer = parts[0].strip(), parts[1].strip()
            if not question or not answer:
                continue

            # Generate format variants
            for fmt, q_text, a_text in categorize_qa(question, answer):
                sid = f"cholec80_v{vid_num}_f{frame_num}_{line_idx}_{fmt}"
                sample = {
                    "id": sid,
                    "answer_format": fmt,
                    "video": video_rel,
                    "video_start": round(max(0, frame_sec - FRAME_WINDOW_SEC), 2),
                    "video_end": round(frame_sec + FRAME_WINDOW_SEC, 2),
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": f"<video>\n{q_text}"},
                        {"from": "gpt", "value": a_text},
                    ],
                }
                samples.append(sample)

print(f"\nGenerated {len(samples)} samples from {len(video_dirs) - skipped} videos")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "cholec80_vqa_train.json")
with open(out_path, "w") as f:
    json.dump(samples, f, ensure_ascii=False)
print(f"Saved → {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

# Stats
from collections import Counter
fmt_counts = Counter(s["answer_format"] for s in samples)
print(f"Format distribution: {dict(fmt_counts)}")
