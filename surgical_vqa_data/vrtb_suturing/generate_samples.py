"""
Generate VQA samples from a single VTRB-Suturing episode.

For each task type, generates:
  - Multiple question phrasings (3-5)
  - Multiple answer format variants (json / nl / single)
  - For JSON answers: chain-of-thought prefix before JSON
  - For descriptions: 3-type LLaVA pattern (simple / detailed / reasoning)

Outputs:
  - samples.json          : all QA pairs in LLaVA format
  - visualizations/       : extracted clips (.mp4) and key-frames (.jpg)
"""

import json, os, subprocess, textwrap, itertools

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATED_BASE = "/home/ubuntu/datasets/vlm/VTRB-Suturing/annotated"
SESSION = "20260328_hyunjun"
EPISODE_IDX = 1
FPS = 20

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.join(OUT_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

MARGIN_BEFORE_SEC = 1.0
MARGIN_AFTER_SEC  = 3.0

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a surgical video analysis assistant specialized in robotic suturing procedures. "
    "When the user asks a question, respond in the exact format requested:\n"
    "- If the question specifies a JSON schema, first provide a brief reasoning, "
    "then output the JSON on a new line.\n"
    "- If the question asks for a single choice, respond with ONLY the chosen option.\n"
    "- If the question is open-ended, respond with a concise natural language description."
)

# ── Load metadata ─────────────────────────────────────────────────────────────
session_dir = os.path.join(ANNOTATED_BASE, SESSION)
video_path = os.path.join(
    session_dir,
    f"videos/chunk-000/laparoscope_camera/episode_{EPISODE_IDX:06d}.mp4",
)

with open(os.path.join(session_dir, "meta", "episodes.jsonl")) as f:
    for line in f:
        ep = json.loads(line.strip())
        if ep["episode_index"] == EPISODE_IDX:
            break

with open(os.path.join(session_dir, "meta", "report.jsonl")) as f:
    report = None
    for line in f:
        r = json.loads(line.strip())
        if r["episode_index"] == EPISODE_IDX:
            report = r
            break

tasks = ep["tasks"]
completions = ep["subtask_completion_steps"]
episode_len = ep["length"]

phases = []
prev = 0
for i, (task, comp) in enumerate(zip(tasks, completions)):
    end = comp if comp is not None else episode_len
    phases.append((prev, end, task, i))
    prev = end

failure_reasons = report["report"] if report else []
failure_subtask_idxs = report["subtask_index"] if report else []
failure_map = {si: r for si, r in zip(failure_subtask_idxs, failure_reasons)}

retry_map = {}
for i in range(1, len(phases)):
    if phases[i][2] == phases[i - 1][2]:
        retry_map[i - 1] = i

print(f"Episode {EPISODE_IDX}: {len(phases)} phases, {episode_len} frames, "
      f"{len(failure_reasons)} failures, retries: {retry_map}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def s2s(step):
    return step / FPS

def phase_clip_range(phase_idx, margin_before=0.0, margin_after=0.0,
                     extend_into_retry=False):
    p = phases[phase_idx]
    start_step, end_step = p[0], p[1]
    if extend_into_retry and phase_idx in retry_map:
        rp = phases[retry_map[phase_idx]]
        end_step = rp[0] + min((rp[1] - rp[0]) // 2, FPS * 5)
    return (max(0.0, s2s(start_step) - margin_before),
            min(s2s(episode_len), s2s(end_step) + margin_after))

def extract_clip(src, out_path, start_sec, end_sec):
    dur = end_sec - start_sec
    subprocess.run([
        "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", src,
        "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-an", "-loglevel", "error", out_path,
    ], check=True)

def extract_frame(src, out_path, time_sec):
    subprocess.run([
        "ffmpeg", "-y", "-ss", f"{time_sec:.3f}", "-i", src,
        "-frames:v", "1", "-q:v", "2", "-loglevel", "error", out_path,
    ], check=True)

def jdump(d):
    return json.dumps(d, ensure_ascii=False)

def make_sample(task_id, media_type, media_file, question, answer,
                video_start=None, video_end=None, answer_format="json"):
    sample = {"id": task_id, "answer_format": answer_format}
    if media_type == "video":
        sample["video"] = os.path.relpath(video_path, OUT_DIR)
        if video_start is not None:
            sample["video_start"] = round(video_start, 2)
        if video_end is not None:
            sample["video_end"] = round(video_end, 2)
        token = "<video>"
    else:
        sample["image"] = os.path.relpath(media_file, OUT_DIR)
        token = "<image>"
    sample["conversations"] = [
        {"from": "system", "value": SYSTEM_PROMPT},
        {"from": "human", "value": f"{token}\n{question}"},
        {"from": "gpt", "value": answer},
    ]
    return sample


# ── Constants ─────────────────────────────────────────────────────────────────
CANONICAL_PHASES = [
    "Left arm: pick up the needle",
    "Left arm: insert the needle into the tissue",
    "Left arm: grasp the needle",
    "Left arm: pull the needle through",
    "Left arm: grasp the thread for pulling",
    "Left arm: pull the thread through",
    "Left arm: grasp the thread for wrapping",
    "Both arms: wrap the thread around the right gripper for the first loop",
    "Both arms: wrap the thread around the right gripper for the second loop",
    "Right arm: grasp the tail of the thread",
    "Both arms: pull the thread to tie the first knot",
    "Both arms: release the grippers",
    "Right arm: grasp the thread for wrapping",
    "Both arms: wrap the thread around the left gripper for the first loop",
    "Left arm: grasp the tail of the thread",
    "Both arms: pull the thread to tie the second knot",
    "Left arm: release the gripper",
    "Left arm: cut the thread",
]
PHASE_LIST = "\n".join(f"  {i}. {p}" for i, p in enumerate(CANONICAL_PHASES))

FAILURE_TYPES = [
    "Bounce off", "Miss target", "Partially observable",
    "Not tightened", "Unwrapped", "Loose", "Stop behavior",
    "Twisted", "Collision", "Open gripper", "Cut thread",
    "Knot untied", "Patient harm",
]
FAILURE_LIST = ", ".join(FAILURE_TYPES)

def arm_of(task_name):
    if task_name.startswith("Both arms"): return "Both arms"
    if task_name.startswith("Left arm"):  return "Left arm"
    return "Right arm"


# ══════════════════════════════════════════════════════════════════════════════
#  Extract visuals (once per clip/image — reused across phrasings & formats)
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting clips and frames...")

# V01: Phase ID video — phase 6
v01_p = phases[6]
v01_s, v01_e = s2s(v01_p[0]), s2s(v01_p[1])
v01_clip = os.path.join(VIZ_DIR, "q01_phase_id_video.mp4")
extract_clip(video_path, v01_clip, v01_s, v01_e)

# V02: Phase ID image — phase 10
v02_p = phases[10]
v02_sec = s2s((v02_p[0] + v02_p[1]) // 2)
v02_img = os.path.join(VIZ_DIR, "q02_phase_id_image.jpg")
extract_frame(video_path, v02_img, v02_sec)

# V03: Progress video — phase 3 at 40%
v03_p = phases[3]
v03_seg = v03_p[1] - v03_p[0]
v03_step = v03_p[0] + int(v03_seg * 0.4)
v03_s = s2s(max(v03_p[0], v03_step - FPS))
v03_e = s2s(min(v03_p[1], v03_step + FPS))
v03_clip = os.path.join(VIZ_DIR, "q03_progress_video.mp4")
extract_clip(video_path, v03_clip, v03_s, v03_e)
v03_pct = int(round((v03_step - v03_p[0]) / v03_seg * 100))

# V04: Progress image — phase 8 at 75%
v04_p = phases[8]
v04_seg = v04_p[1] - v04_p[0]
v04_step = v04_p[0] + int(v04_seg * 0.75)
v04_sec = s2s(v04_step)
v04_img = os.path.join(VIZ_DIR, "q04_progress_image.jpg")
extract_frame(video_path, v04_img, v04_sec)
v04_pct = int(round((v04_step - v04_p[0]) / v04_seg * 100))

# V05: Next phase — phase 4, last 25%
v05_p = phases[4]
v05_seg = v05_p[1] - v05_p[0]
v05_s = s2s(v05_p[0] + int(v05_seg * 0.75))
v05_e = s2s(v05_p[1])
v05_clip = os.path.join(VIZ_DIR, "q05_next_phase.mp4")
extract_clip(video_path, v05_clip, v05_s, v05_e)

# V06: Prev phase — phase 9
v06_p = phases[9]
v06_s, v06_e = s2s(v06_p[0]), s2s(v06_p[1])
v06_clip = os.path.join(VIZ_DIR, "q06_prev_phase.mp4")
extract_clip(video_path, v06_clip, v06_s, v06_e)

# V07: Ordering image — phase 12
v07_p = phases[12]
v07_sec = s2s((v07_p[0] + v07_p[1]) // 2)
v07_img = os.path.join(VIZ_DIR, "q07_ordering.jpg")
extract_frame(video_path, v07_img, v07_sec)

# V08a: Success fail — phase 13 (with retry margin)
v08a_s, v08a_e = phase_clip_range(13, MARGIN_BEFORE_SEC, MARGIN_AFTER_SEC, True)
v08a_clip = os.path.join(VIZ_DIR, "q08_success_fail.mp4")
extract_clip(video_path, v08a_clip, v08a_s, v08a_e)

# V08b: Success ok — phase 10
v08b_s, v08b_e = phase_clip_range(10, 0.5, 0.5)
v08b_clip = os.path.join(VIZ_DIR, "q08_success_ok.mp4")
extract_clip(video_path, v08b_clip, v08b_s, v08b_e)

# V09: Failure reason — phase 15 (with retry margin)
v09_s, v09_e = phase_clip_range(15, MARGIN_BEFORE_SEC, MARGIN_AFTER_SEC, True)
v09_clip = os.path.join(VIZ_DIR, "q09_failure_reason.mp4")
extract_clip(video_path, v09_clip, v09_s, v09_e)

# V10: Episode success — full episode
v10_s, v10_e = 0.0, s2s(episode_len)
v10_clip = os.path.join(VIZ_DIR, "q10_episode_success.mp4")
extract_clip(video_path, v10_clip, v10_s, v10_e)

# V11: Completion — phase 7, last 10%
v11_p = phases[7]
v11_seg = v11_p[1] - v11_p[0]
v11_s = s2s(v11_p[0] + int(v11_seg * 0.9))
v11_e = s2s(v11_p[1])
v11_clip = os.path.join(VIZ_DIR, "q11_completion_yes.mp4")
extract_clip(video_path, v11_clip, v11_s, v11_e)

# V12: Arm ID image — phase 7
v12_sec = s2s((v11_p[0] + v11_p[1]) // 2)
v12_img = os.path.join(VIZ_DIR, "q12_arm_id.jpg")
extract_frame(video_path, v12_img, v12_sec)

# V13: Description — phase 1
v13_p = phases[1]
v13_s, v13_e = s2s(v13_p[0]), s2s(v13_p[1])
v13_clip = os.path.join(VIZ_DIR, "q13_description.mp4")
extract_clip(video_path, v13_clip, v13_s, v13_e)

# V14: Failure localization — phases 12-15 with margins
v14_s = max(0.0, s2s(phases[12][0]) - MARGIN_BEFORE_SEC)
v14_e = min(s2s(episode_len), s2s(phases[15][1]) + MARGIN_AFTER_SEC)
v14_clip = os.path.join(VIZ_DIR, "q14_failure_localization.mp4")
extract_clip(video_path, v14_clip, v14_s, v14_e)

print("Visual extraction done.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Generate all QA pairs: phrasings × format variants
# ══════════════════════════════════════════════════════════════════════════════
samples = []
sid = [0]  # mutable counter

def add(task_base, media_type, media_file, question, answer,
        vs=None, ve=None, answer_format="json"):
    sid[0] += 1
    samples.append(make_sample(
        f"{task_base}_{sid[0]:03d}", media_type, media_file,
        question, answer, vs, ve, answer_format,
    ))

# Helper: phase index in canonical list (or -1 if not found)
def pidx(name):
    return CANONICAL_PHASES.index(name) if name in CANONICAL_PHASES else -1


# ──────────────────────────────────────────────────────────────────────────────
#  Q1: Phase ID — VIDEO (5 phrasings × 3 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name = v01_p[2]
p_idx = pidx(p_name)

Q1_PHRASINGS = [
    "What surgical phase is being performed in this video clip?",
    "Identify the current phase of the suturing procedure shown in this video.",
    "Which step of the suturing procedure is the robot performing?",
    "What is the surgeon doing in this clip? Identify the surgical phase.",
    "Determine the surgical phase depicted in this video segment.",
]

for q in Q1_PHRASINGS:
    # JSON
    add("phase_id_video", "video", v01_clip,
        f"{q}\nChoose from:\n{PHASE_LIST}\n\n"
        f"Respond with JSON: {{\"phase\": \"<name>\", \"phase_index\": <int>}}",
        f"The gripper is grasping the thread in preparation for wrapping.\n"
        f"{jdump({'phase': p_name, 'phase_index': p_idx})}",
        v01_s, v01_e, "json")
    # NL
    add("phase_id_video", "video", v01_clip,
        f"{q}\nChoose from:\n{PHASE_LIST}\n\nRespond in a complete sentence.",
        f"The current surgical phase is '{p_name}' (phase {p_idx} of {len(CANONICAL_PHASES)}). "
        f"The left arm is grasping the thread to prepare for the wrapping step.",
        v01_s, v01_e, "nl")
    # Single
    add("phase_id_video", "video", v01_clip,
        f"{q}\nChoose from:\n{PHASE_LIST}\n\nRespond with ONLY the phase name.",
        p_name,
        v01_s, v01_e, "single")

# ──────────────────────────────────────────────────────────────────────────────
#  Q2: Phase ID — IMAGE (5 phrasings × 3 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name = v02_p[2]
p_idx = pidx(p_name)

Q2_PHRASINGS = [
    "What surgical phase is being performed in this frame?",
    "Identify the current suturing phase from this image.",
    "Which phase of the procedure does this frame depict?",
    "Based on this image, what surgical step is underway?",
    "Determine the phase of the suturing procedure shown here.",
]

for q in Q2_PHRASINGS:
    add("phase_id_image", "image", v02_img,
        f"{q}\nChoose from:\n{PHASE_LIST}\n\n"
        f"Respond with JSON: {{\"phase\": \"<name>\", \"phase_index\": <int>}}",
        f"Both grippers are pulling the thread in opposite directions to tighten.\n"
        f"{jdump({'phase': p_name, 'phase_index': p_idx})}",
        answer_format="json")
    add("phase_id_image", "image", v02_img,
        f"{q}\nChoose from:\n{PHASE_LIST}\n\nRespond in a complete sentence.",
        f"The current surgical phase is '{p_name}' (phase {p_idx} of {len(CANONICAL_PHASES)}). "
        f"Both arms are actively pulling the thread to form the first knot.",
        answer_format="nl")
    add("phase_id_image", "image", v02_img,
        f"{q}\nChoose from:\n{PHASE_LIST}\n\nRespond with ONLY the phase name.",
        p_name,
        answer_format="single")

# ──────────────────────────────────────────────────────────────────────────────
#  Q3: Progress — VIDEO (4 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name = v03_p[2]
next_name = phases[4][2]

Q3_PHRASINGS = [
    "Analyze this video clip. Identify the current phase, estimate progress (0-100%), and predict the next phase.",
    "What phase is this, how far along is it, and what comes next?",
    "Determine the surgical phase, its completion percentage, and the upcoming phase.",
    "Assess the current state: phase name, progress percentage, and next expected phase.",
]

for q in Q3_PHRASINGS:
    # JSON (with CoT prefix)
    add("progress_video", "video", v03_clip,
        f"{q}\n\nRespond with JSON: {{\"phase\": \"<name>\", \"progress\": <int 0-100>, \"next_phase\": \"<name>\"}}",
        f"The needle is partially through the tissue, roughly two-fifths of the way.\n"
        f"{jdump({'phase': p_name, 'progress': v03_pct, 'next_phase': p_name})}",
        v03_s, v03_e, "json")
    # NL
    add("progress_video", "video", v03_clip,
        f"{q}\n\nRespond in a complete sentence.",
        f"The current phase is '{p_name}' at approximately {v03_pct}% progress. "
        f"The needle is being pulled through the tissue but is not yet fully extracted. "
        f"Since the phase is still in progress, the next step remains '{p_name}'.",
        v03_s, v03_e, "nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q4: Progress — IMAGE (4 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name = v04_p[2]
next_name = phases[9][2]

Q4_PHRASINGS = [
    "Analyze this frame. Identify the current phase, estimate progress (0-100%), and predict the next phase.",
    "What phase does this frame show, how far along, and what follows?",
    "From this image, determine the phase, progress percentage, and next phase.",
    "Assess the surgical state in this frame: phase, progress, and upcoming phase.",
]

for q in Q4_PHRASINGS:
    add("progress_image", "image", v04_img,
        f"{q}\n\nRespond with JSON: {{\"phase\": \"<name>\", \"progress\": <int 0-100>, \"next_phase\": \"<name>\"}}",
        f"The thread is nearly fully wrapped around the gripper for the second loop.\n"
        f"{jdump({'phase': p_name, 'progress': v04_pct, 'next_phase': next_name})}",
        answer_format="json")
    add("progress_image", "image", v04_img,
        f"{q}\n\nRespond in a complete sentence.",
        f"The current phase is '{p_name}' at approximately {v04_pct}% progress. "
        f"The wrapping is nearly complete. The next phase will be '{next_name}'.",
        answer_format="nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q5: Next phase — VIDEO (4 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
cur_name = v05_p[2]
nxt_name = phases[5][2]

Q5_PHRASINGS = [
    "Based on this video, identify the current phase and predict the next one.",
    "What phase is shown, and what phase will follow?",
    "Determine the current and upcoming surgical phase from this clip.",
    "What is happening now and what should happen next in this procedure?",
]

for q in Q5_PHRASINGS:
    add("next_phase", "video", v05_clip,
        f"{q}\nIf the current phase is still in progress, set next_phase to the current phase.\n"
        f"Choices:\n{PHASE_LIST}\n\n"
        f"Respond with JSON: {{\"current_phase\": \"<name>\", \"next_phase\": \"<name>\", \"is_transitioning\": <bool>}}",
        f"The thread grasping is nearly complete, approaching transition.\n"
        f"{jdump({'current_phase': cur_name, 'next_phase': nxt_name, 'is_transitioning': True})}",
        v05_s, v05_e, "json")
    add("next_phase", "video", v05_clip,
        f"{q}\nIf the current phase is still in progress, the next phase is the same.\n"
        f"Choices:\n{PHASE_LIST}\n\nRespond in a complete sentence.",
        f"The current phase is '{cur_name}' and it is nearing completion. "
        f"The next phase will be '{nxt_name}', where the thread is pulled through the tissue.",
        v05_s, v05_e, "nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q6: Previous phase — VIDEO (4 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
cur_name = v06_p[2]
prv_name = phases[8][2]

Q6_PHRASINGS = [
    "What phase was performed immediately before the one shown in this video?",
    "Identify the current phase and the one that preceded it.",
    "What surgical step came just before the current one?",
    "Determine the current and previous phase from this clip.",
]

for q in Q6_PHRASINGS:
    add("prev_phase", "video", v06_clip,
        f"{q}\nChoices:\n{PHASE_LIST}\n\n"
        f"Respond with JSON: {{\"current_phase\": \"<name>\", \"previous_phase\": \"<name>\"}}",
        f"The right arm is reaching for the thread tail after wrapping.\n"
        f"{jdump({'current_phase': cur_name, 'previous_phase': prv_name})}",
        v06_s, v06_e, "json")
    add("prev_phase", "video", v06_clip,
        f"{q}\nChoices:\n{PHASE_LIST}\n\nRespond in a complete sentence.",
        f"The current phase is '{cur_name}'. The previous phase was '{prv_name}', "
        f"where the thread was wrapped around the right gripper for the second loop.",
        v06_s, v06_e, "nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q7: Phase ordering — IMAGE (3 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name = v07_p[2]
remaining = len(phases) - 1 - 12
total = len(phases)

Q7_PHRASINGS = [
    "Identify the current phase and its position in the overall suturing procedure.",
    "What phase is this, and how far along is the overall procedure?",
    "Determine the current phase and whether this is early, middle, or late in the procedure.",
]

for q in Q7_PHRASINGS:
    add("ordering", "image", v07_img,
        f"{q}\n\nRespond with JSON: {{\"phase\": \"<name>\", \"phase_index\": <int>, "
        f"\"total_phases\": <int>, \"stage\": \"early|middle|late\", \"phases_remaining\": <int>}}",
        f"This is a wrapping phase near the end of the procedure.\n"
        f"{jdump({'phase': p_name, 'phase_index': 12, 'total_phases': total, 'stage': 'late', 'phases_remaining': remaining})}",
        answer_format="json")
    add("ordering", "image", v07_img,
        f"{q}\n\nRespond in a complete sentence.",
        f"The current phase is '{p_name}' (phase 13 of {total}), which is in the late stage "
        f"of the procedure. There are {remaining} phases remaining.",
        answer_format="nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q8a: Success detection FAIL — VIDEO (4 phrasings × 3 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name = phases[13][2]
fail_reason = failure_map[13]

Q8_PHRASINGS = [
    f"Was the phase \"{p_name}\" completed successfully in this video?",
    f"Did the robot successfully complete \"{p_name}\"?",
    f"Assess whether the phase \"{p_name}\" was performed without errors.",
    f"Evaluate the outcome of the phase \"{p_name}\" shown in this clip.",
]

for q in Q8_PHRASINGS:
    add("success_fail", "video", v08a_clip,
        f"{q}\n\nRespond with JSON: {{\"success\": <bool>, \"failure_reason\": \"<reason or null>\"}}",
        f"The gripper missed the intended wrapping target and had to retry.\n"
        f"{jdump({'success': False, 'failure_reason': fail_reason})}",
        v08a_s, v08a_e, "json")
    add("success_fail", "video", v08a_clip,
        f"{q}\n\nRespond in a complete sentence.",
        f"No, the phase '{p_name}' was not completed successfully. "
        f"The failure reason was '{fail_reason}' — the instrument missed its intended target, "
        f"requiring the phase to be retried.",
        v08a_s, v08a_e, "nl")
    add("success_fail", "video", v08a_clip,
        f"{q}\n\nAnswer only: Yes or No",
        "No",
        v08a_s, v08a_e, "single")

# ──────────────────────────────────────────────────────────────────────────────
#  Q8b: Success detection OK — VIDEO (4 phrasings × 3 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name_ok = phases[10][2]

for q_template in Q8_PHRASINGS:
    q = q_template.replace(phases[13][2], p_name_ok)
    add("success_ok", "video", v08b_clip,
        f"{q}\n\nRespond with JSON: {{\"success\": <bool>, \"failure_reason\": \"<reason or null>\"}}",
        f"The thread was pulled smoothly to tie the knot without issues.\n"
        f"{jdump({'success': True, 'failure_reason': None})}",
        v08b_s, v08b_e, "json")
    add("success_ok", "video", v08b_clip,
        f"{q}\n\nRespond in a complete sentence.",
        f"Yes, the phase '{p_name_ok}' was completed successfully. "
        f"Both arms pulled the thread to tie the first knot without any errors.",
        v08b_s, v08b_e, "nl")
    add("success_ok", "video", v08b_clip,
        f"{q}\n\nAnswer only: Yes or No",
        "Yes",
        v08b_s, v08b_e, "single")

# ──────────────────────────────────────────────────────────────────────────────
#  Q9: Failure reason — VIDEO (4 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name = phases[15][2]
fail_reason = failure_map[15]
was_retried = 15 in retry_map

Q9_PHRASINGS = [
    f"The phase \"{p_name}\" failed. What was the failure reason?",
    f"This phase was not successful. Identify the type of failure that occurred.",
    f"What went wrong during \"{p_name}\"? Classify the failure.",
    f"Diagnose the failure in this phase. What type of error occurred?",
]

for q in Q9_PHRASINGS:
    add("failure_reason", "video", v09_clip,
        f"{q}\nChoose from: {FAILURE_LIST}\n\n"
        f"Respond with JSON: {{\"phase\": \"<name>\", \"failure_reason\": \"<reason>\", \"was_retried\": <bool>}}",
        f"The gripper reached for the thread tail but did not make contact at the right position.\n"
        f"{jdump({'phase': p_name, 'failure_reason': fail_reason, 'was_retried': was_retried})}",
        v09_s, v09_e, "json")
    add("failure_reason", "video", v09_clip,
        f"{q}\nChoose from: {FAILURE_LIST}\n\nRespond in a complete sentence.",
        f"The failure during '{p_name}' was '{fail_reason}'. "
        f"The instrument did not reach the correct target position. "
        f"{'The phase was retried immediately after.' if was_retried else 'The procedure continued.'}",
        v09_s, v09_e, "nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q10: Episode-level success — VIDEO (3 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
fail_details = [{"phase": tasks[si], "phase_index": si, "failure_reason": r}
                for si, r in zip(failure_subtask_idxs, failure_reasons)]

Q10_PHRASINGS = [
    "Was this entire suturing attempt successful? If not, list each failure.",
    "Evaluate the overall outcome of this suturing procedure.",
    "Assess this complete suturing episode: was it successful, and what errors occurred?",
]

fail_desc = "; ".join(f"'{r}' during '{tasks[si]}'" for si, r in zip(failure_subtask_idxs, failure_reasons))

for q in Q10_PHRASINGS:
    add("episode_success", "video", v10_clip,
        f"{q}\n\nRespond with JSON: {{\"success\": <bool>, \"failures\": [{{\"phase\": \"<name>\", \"phase_index\": <int>, \"failure_reason\": \"<reason>\"}}]}}",
        f"Multiple errors occurred: partially observable pull-through, missed targets during wrapping and grasping.\n"
        f"{jdump({'success': False, 'failures': fail_details})}",
        v10_s, v10_e, "json")
    add("episode_success", "video", v10_clip,
        f"{q}\n\nRespond in a complete sentence.",
        f"The suturing attempt was not fully successful. "
        f"Failures occurred: {fail_desc}. "
        f"Despite these errors, the procedure was completed with retries.",
        v10_s, v10_e, "nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q11: Completion detection — VIDEO (4 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
p_name = v11_p[2]

Q11_PHRASINGS = [
    f"Has the phase \"{p_name}\" been completed by the end of this clip?",
    f"Is the phase \"{p_name}\" finished in this video?",
    f"By the end of this clip, has \"{p_name}\" reached completion?",
    f"Determine whether \"{p_name}\" is complete or still in progress.",
]

for q in Q11_PHRASINGS:
    add("completion", "video", v11_clip,
        f"{q}\n\nAnswer only: Yes or No",
        "Yes",
        v11_s, v11_e, "single")
    add("completion", "video", v11_clip,
        f"{q}\n\nRespond in a complete sentence.",
        f"Yes, the phase '{p_name}' has been completed by the end of this clip. "
        f"The thread has been fully wrapped around the gripper for the first loop.",
        v11_s, v11_e, "nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q12: Arm identification — IMAGE (4 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
arm = arm_of(v11_p[2])

Q12_PHRASINGS = [
    "Which arm(s) are actively performing the surgical action in this frame?",
    "Identify which robotic arm(s) are engaged in the current action.",
    "Is the left arm, right arm, or both arms active in this image?",
    "Determine which arm(s) are performing the task shown here.",
]

for q in Q12_PHRASINGS:
    add("arm_id", "image", v12_img,
        f"{q}\nChoose from: Left arm, Right arm, Both arms\n\nAnswer only with your choice.",
        arm,
        answer_format="single")
    add("arm_id", "image", v12_img,
        f"{q}\nChoose from: Left arm, Right arm, Both arms\n\nRespond in a complete sentence.",
        f"{arm} are actively performing the surgical action. "
        f"The current phase is '{v11_p[2]}', which requires coordinated bimanual manipulation.",
        answer_format="nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q13: Action description — VIDEO (4 phrasings × 3 LLaVA types)
# ──────────────────────────────────────────────────────────────────────────────
p_name = v13_p[2]

Q13_PHRASINGS = [
    "Describe what is happening in this surgical video clip.",
    "What surgical action is being performed in this video?",
    "Explain the procedure shown in this clip.",
    "What is the robot doing in this video?",
]

# Answers for 3 LLaVA conversation types
A13_SIMPLE = (
    f"The left arm is inserting the needle into the tissue. "
    f"This is the '{p_name}' phase of the suturing procedure."
)
A13_DETAIL = (
    f"In this clip, the left robotic arm is performing the '{p_name}' phase. "
    f"The needle holder grips the needle at its mid-point and drives it through "
    f"the tissue in a curved trajectory, following the needle's curvature. "
    f"The entry point is on the near side of the tissue pad, and the needle tip "
    f"must emerge from the far side at the correct depth and angle. "
    f"This is an early phase (phase 2 of {len(CANONICAL_PHASES)}) in the suturing sequence. "
    f"The right arm remains stationary, ready to assist in subsequent phases."
)
A13_REASONING = (
    f"The video shows the '{p_name}' phase. The needle is being driven through the "
    f"tissue following its natural curvature, which is essential for proper suture placement. "
    f"The insertion angle and depth are critical — too shallow risks tissue tearing, "
    f"while too deep could damage underlying structures. The left arm's pronation "
    f"suggests it is rotating the needle holder to follow the curved path. "
    f"Based on the needle's current position (partially through), this phase is "
    f"approximately 40-50% complete. After full insertion, the next phase will be "
    f"grasping the needle on the other side to pull it through."
)

for q in Q13_PHRASINGS:
    add("description", "video", v13_clip,
        f"{q}\n\nRespond in 1-2 sentences.",
        A13_SIMPLE, v13_s, v13_e, "nl")
    add("description", "video", v13_clip,
        f"{q}\n\nProvide a detailed description of the surgical actions, "
        f"instrument positions, and procedural context.",
        A13_DETAIL, v13_s, v13_e, "nl")
    add("description", "video", v13_clip,
        f"{q}\n\nAnalyze the technique, assess quality, and reason about "
        f"what should happen next.",
        A13_REASONING, v13_s, v13_e, "nl")

# ──────────────────────────────────────────────────────────────────────────────
#  Q14: Failure localization — VIDEO (3 phrasings × 2 formats)
# ──────────────────────────────────────────────────────────────────────────────
fail_phase = phases[13][2]
fail_r = failure_map[13]

Q14_PHRASINGS = [
    "This clip spans multiple phases. A failure occurred and was retried. Identify the failure.",
    "Locate the failure in this multi-phase clip. Which phase failed and why?",
    "Analyze this clip for errors. Identify the failed phase, failure type, and whether it was retried.",
]

for q in Q14_PHRASINGS:
    add("failure_loc", "video", v14_clip,
        f"{q}\n\nRespond with JSON: {{\"failed_phase\": \"<name>\", \"failure_reason\": \"<reason>\", "
        f"\"was_retried\": <bool>, \"retry_successful\": <bool>}}",
        f"During wrapping, the gripper missed the target, requiring a retry attempt.\n"
        f"{jdump({'failed_phase': fail_phase, 'failure_reason': fail_r, 'was_retried': True, 'retry_successful': True})}",
        v14_s, v14_e, "json")
    add("failure_loc", "video", v14_clip,
        f"{q}\n\nRespond in a complete sentence.",
        f"The failure occurred during '{fail_phase}': the robot experienced a '{fail_r}' error "
        f"where the instrument did not reach the intended position. "
        f"The phase was retried immediately and the retry was successful.",
        v14_s, v14_e, "nl")


# ══════════════════════════════════════════════════════════════════════════════
#  Save
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(OUT_DIR, "samples.json")
with open(out_path, "w") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

# ── Summary ───────────────────────────────────────────────────────────────────
from collections import Counter
fmt_counts = Counter(s["answer_format"] for s in samples)
task_counts = Counter(s["id"].rsplit("_", 1)[0] for s in samples)

print(f"{'='*70}")
print(f"Generated {len(samples)} samples → {out_path}")
print(f"\nFormat distribution:")
for fmt, cnt in fmt_counts.most_common():
    print(f"  {fmt:>6}: {cnt:3d}  ({cnt/len(samples)*100:.0f}%)")
print(f"\nTask distribution:")
for task, cnt in sorted(task_counts.items()):
    print(f"  {task:<30}: {cnt:3d}")
