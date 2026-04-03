"""
Generate full VTRB-Suturing VQA dataset across ALL annotated episodes.

Produces:
  - vrtb_suturing_train.json  (LLaVA format, ~700K QA pairs)
  - vrtb_suturing_val.json    (50 samples, covering all task types)
"""

import json, os, glob, random, textwrap, re
from collections import Counter, defaultdict

random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATED_BASE = "/home/ubuntu/datasets/vlm/VTRB-Suturing/annotated"
FPS = 20
MARGIN_BEFORE = 1.0
MARGIN_AFTER = 3.0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = (
    "You are a surgical video analysis assistant specialized in robotic suturing procedures. "
    "When the user asks a question, respond in the exact format requested:\n"
    "- If the question specifies a JSON schema, first provide a brief reasoning, "
    "then output the JSON on a new line.\n"
    "- If the question asks for a single choice, respond with ONLY the chosen option.\n"
    "- If the question is open-ended, respond with a concise natural language description."
)

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

# Normalize failure labels
FAILURE_NORM = {
    "Bounceoff": "Bounce off", "bounce off": "Bounce off",
    "Patient_Harm": "Patient harm", "Patient_harm": "Patient harm",
    "Human_harm": "Patient harm", "Harm_Patient": "Patient harm",
    "Loose_knot": "Loose", "no_tight": "Not tightened",
    "pull_hard": "Excessive force", "wrapping_direction": "Wrong direction",
    "wrapping_pull_too_much": "Excessive force",
    "Wrapped_second_loop": "Wrong wrapping", "Knot_untied": "Knot untied",
    "Cut_thread": "Cut thread", "Open_gripper": "Open gripper",
    "Tied_3times": "Extra tie",
}


def norm_failure(r):
    return FAILURE_NORM.get(r, r)


def jdump(d):
    return json.dumps(d, ensure_ascii=False)


def pidx(name):
    return CANONICAL_PHASES.index(name) if name in CANONICAL_PHASES else -1


def arm_of(name):
    if name.startswith("Both arms"): return "Both arms"
    if name.startswith("Left arm"):  return "Left arm"
    return "Right arm"


# ── Question templates ────────────────────────────────────────────────────────
PHASE_ID_QS = [
    "What surgical phase is being performed?",
    "Identify the current phase of the suturing procedure.",
    "Which step of the suturing procedure is the robot performing?",
    "What is the surgeon doing? Identify the surgical phase.",
    "Determine the surgical phase depicted.",
]

PROGRESS_QS = [
    "Identify the current phase, estimate progress (0-100%), and predict the next phase.",
    "What phase is this, how far along is it, and what comes next?",
    "Determine the surgical phase, its completion percentage, and the upcoming phase.",
    "Assess the current state: phase name, progress percentage, and next expected phase.",
]

NEXT_QS = [
    "Identify the current phase and predict the next one.",
    "What phase is shown, and what phase will follow?",
    "Determine the current and upcoming surgical phase.",
    "What is happening now and what should happen next?",
]

PREV_QS = [
    "What phase was performed immediately before the one shown?",
    "Identify the current phase and the one that preceded it.",
    "What surgical step came just before the current one?",
    "Determine the current and previous phase.",
]

ORDERING_QS = [
    "Identify the current phase and its position in the overall procedure.",
    "What phase is this, and how far along is the overall procedure?",
    "Determine whether this is early, middle, or late in the procedure.",
]

SUCCESS_QS = [
    "Was this phase completed successfully?",
    "Did the robot successfully complete this phase?",
    "Assess whether this phase was performed without errors.",
    "Evaluate the outcome of this phase.",
]

FAILURE_QS = [
    "This phase failed. What was the failure reason?",
    "This phase was not successful. Identify the failure type.",
    "What went wrong during this phase? Classify the failure.",
    "Diagnose the failure. What type of error occurred?",
]

EPISODE_QS = [
    "Was this entire suturing attempt successful? If not, list each failure.",
    "Evaluate the overall outcome of this suturing procedure.",
    "Assess this suturing episode: was it successful, and what errors occurred?",
]

COMPLETION_QS = [
    "Has this phase been completed by the end of this clip?",
    "Is this phase finished?",
    "By the end of this clip, has this phase reached completion?",
    "Determine whether this phase is complete or still in progress.",
]

ARM_QS = [
    "Which arm(s) are actively performing the surgical action?",
    "Identify which robotic arm(s) are engaged.",
    "Is the left arm, right arm, or both arms active?",
    "Determine which arm(s) are performing the task.",
]

DESC_QS = [
    "Describe what is happening in this surgical clip.",
    "What surgical action is being performed?",
    "Explain the procedure shown.",
    "What is the robot doing?",
]


# ── Load all episodes ─────────────────────────────────────────────────────────
all_episodes = []

for session_dir in sorted(glob.glob(f"{ANNOTATED_BASE}/*/")):
    session = os.path.basename(session_dir.rstrip("/"))
    ep_file = os.path.join(session_dir, "meta", "episodes.jsonl")
    report_file = os.path.join(session_dir, "meta", "report.jsonl")
    if not os.path.exists(ep_file):
        continue

    reports = {}
    if os.path.exists(report_file):
        with open(report_file) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                reports[r["episode_index"]] = r

    with open(ep_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            ep = json.loads(line)
            ep_idx = ep["episode_index"]

            # Build phases
            phases = []
            prev = 0
            for i, (task, comp) in enumerate(zip(ep["tasks"], ep["subtask_completion_steps"])):
                end = comp if comp is not None else ep["length"]
                phases.append((prev, end, task, i))
                prev = end

            # Failure map
            report = reports.get(ep_idx, {})
            fail_map = {}
            for si, r in zip(report.get("subtask_index", []), report.get("report", [])):
                fail_map[si] = norm_failure(r)

            # Retry map
            retry_map = {}
            for i in range(1, len(phases)):
                if phases[i][2] == phases[i-1][2]:
                    retry_map[i-1] = i

            video_path = os.path.join(
                session_dir,
                f"videos/chunk-000/laparoscope_camera/episode_{ep_idx:06d}.mp4",
            )

            all_episodes.append({
                "session": session,
                "episode_index": ep_idx,
                "length": ep["length"],
                "phases": phases,
                "fail_map": fail_map,
                "retry_map": retry_map,
                "video_path": video_path,
                "report": report,
            })

print(f"Loaded {len(all_episodes)} episodes from {len(glob.glob(f'{ANNOTATED_BASE}/*/'))} sessions")


# ── Generate QA pairs ─────────────────────────────────────────────────────────
samples = []
task_type_counts = Counter()
sid_counter = [0]


def s2s(step):
    return step / FPS


def clip_range(ep, phase_idx, mb=0, ma=0, extend_retry=False):
    phases = ep["phases"]
    p = phases[phase_idx]
    start, end = p[0], p[1]
    if extend_retry and phase_idx in ep["retry_map"]:
        rp = phases[ep["retry_map"][phase_idx]]
        end = rp[0] + min((rp[1] - rp[0]) // 2, FPS * 5)
    return (max(0.0, s2s(start) - mb), min(s2s(ep["length"]), s2s(end) + ma))


def add(task_type, ep, question, answer, vs, ve, fmt, media="video"):
    sid_counter[0] += 1
    task_type_counts[task_type] += 1
    video_rel = os.path.relpath(ep["video_path"], OUT_DIR)
    sample = {
        "id": f"{task_type}_{sid_counter[0]:07d}",
        "answer_format": fmt,
        "video": video_rel,
        "video_start": round(vs, 2),
        "video_end": round(ve, 2),
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"<video>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    samples.append(sample)


print("Generating QA pairs...")

for ep_i, ep in enumerate(all_episodes):
    if ep_i % 50 == 0:
        print(f"  Episode {ep_i}/{len(all_episodes)} ({len(samples)} samples so far)")

    phases = ep["phases"]
    n_phases = len(phases)
    fail_map = ep["fail_map"]

    for pi, (start, end, task, _) in enumerate(phases):
        seg_len = end - start
        if seg_len < 2:
            continue

        vs, ve = s2s(start), s2s(end)
        p_idx = pidx(task)
        arm = arm_of(task)
        is_failed = pi in fail_map

        # Pick one random phrasing per format variant (not all 5 — that's for full scale)
        # Using 2 phrasings per format to keep ~700K target
        q_phase = random.choice(PHASE_ID_QS)
        q_prog = random.choice(PROGRESS_QS)

        # ── Phase ID (video) ─────────────────────────────────────────
        # JSON
        add("phase_id", ep,
            f"{q_phase}\nChoose from:\n{PHASE_LIST}\n\n"
            f"Respond with JSON: {{\"phase\": \"<name>\", \"phase_index\": <int>}}",
            f"{arm} is performing {task.split(': ',1)[-1]}.\n"
            f"{jdump({'phase': task, 'phase_index': p_idx})}",
            vs, ve, "json")
        # NL
        add("phase_id", ep,
            f"{q_phase}\nChoose from:\n{PHASE_LIST}\n\nRespond in a complete sentence.",
            f"The current surgical phase is '{task}'"
            f"{f' (phase {p_idx} of {len(CANONICAL_PHASES)})' if p_idx >= 0 else ''}.",
            vs, ve, "nl")
        # Single
        add("phase_id", ep,
            f"{q_phase}\nChoose from:\n{PHASE_LIST}\n\nRespond with ONLY the phase name.",
            task, vs, ve, "single")

        # ── Progress (3 sample points: early/mid/late) ───────────────
        for pct_target, label in [(15, "early"), (50, "mid"), (85, "late")]:
            sample_step = start + int(seg_len * pct_target / 100)
            pct = pct_target
            win_s = s2s(max(start, sample_step - FPS))
            win_e = s2s(min(end, sample_step + FPS))

            next_phase = phases[pi+1][2] if pi+1 < n_phases else task

            # JSON
            add("progress", ep,
                f"{q_prog}\n\nRespond with JSON: "
                f"{{\"phase\": \"<name>\", \"progress\": <int 0-100>, \"next_phase\": \"<name>\"}}",
                f"The phase is at the {label} stage.\n"
                f"{jdump({'phase': task, 'progress': pct, 'next_phase': task if pct < 80 else next_phase})}",
                win_s, win_e, "json")
            # NL
            add("progress", ep,
                f"{q_prog}\n\nRespond in a complete sentence.",
                f"The current phase is '{task}' at approximately {pct}% progress. "
                f"{'The next phase will be ' + repr(next_phase) + '.' if pct >= 80 else 'The phase is still in progress.'}",
                win_s, win_e, "nl")

        # ── Next phase ───────────────────────────────────────────────
        if pi + 1 < n_phases:
            q = random.choice(NEXT_QS)
            nxt = phases[pi+1][2]
            # Clip from last 25%
            late_s = s2s(start + int(seg_len * 0.75))
            add("next_phase", ep,
                f"{q}\nChoices:\n{PHASE_LIST}\n\n"
                f"Respond with JSON: {{\"current_phase\": \"<name>\", \"next_phase\": \"<name>\", \"is_transitioning\": <bool>}}",
                f"Approaching the end of {task.split(': ',1)[-1]}.\n"
                f"{jdump({'current_phase': task, 'next_phase': nxt, 'is_transitioning': True})}",
                late_s, ve, "json")
            add("next_phase", ep,
                f"{q}\nChoices:\n{PHASE_LIST}\n\nRespond in a complete sentence.",
                f"The current phase is '{task}' nearing completion. The next phase will be '{nxt}'.",
                late_s, ve, "nl")

        # ── Previous phase ───────────────────────────────────────────
        if pi > 0:
            q = random.choice(PREV_QS)
            prv = phases[pi-1][2]
            add("prev_phase", ep,
                f"{q}\nChoices:\n{PHASE_LIST}\n\n"
                f"Respond with JSON: {{\"current_phase\": \"<name>\", \"previous_phase\": \"<name>\"}}",
                f"This phase follows {prv.split(': ',1)[-1]}.\n"
                f"{jdump({'current_phase': task, 'previous_phase': prv})}",
                vs, ve, "json")
            add("prev_phase", ep,
                f"{q}\nChoices:\n{PHASE_LIST}\n\nRespond in a complete sentence.",
                f"The current phase is '{task}'. The previous phase was '{prv}'.",
                vs, ve, "nl")

        # ── Ordering ─────────────────────────────────────────────────
        q = random.choice(ORDERING_QS)
        remaining = n_phases - 1 - pi
        stage = "early" if pi < n_phases / 3 else ("middle" if pi < 2 * n_phases / 3 else "late")
        mid_s = s2s((start + end) // 2)
        add("ordering", ep,
            f"{q}\n\nRespond with JSON: {{\"phase\": \"<name>\", \"phase_index\": <int>, "
            f"\"total_phases\": <int>, \"stage\": \"early|middle|late\", \"phases_remaining\": <int>}}",
            f"This is a {stage}-stage phase.\n"
            f"{jdump({'phase': task, 'phase_index': pi, 'total_phases': n_phases, 'stage': stage, 'phases_remaining': remaining})}",
            vs, ve, "json")
        add("ordering", ep,
            f"{q}\n\nRespond in a complete sentence.",
            f"The current phase is '{task}' (phase {pi+1} of {n_phases}), {stage} stage. "
            f"{remaining} phases remaining.",
            vs, ve, "nl")

        # ── Success detection ────────────────────────────────────────
        q = random.choice(SUCCESS_QS)
        if is_failed:
            reason = fail_map[pi]
            fvs, fve = clip_range(ep, pi, MARGIN_BEFORE, MARGIN_AFTER, True)
            add("success", ep,
                f"{q}\n\nRespond with JSON: {{\"success\": <bool>, \"failure_reason\": \"<reason or null>\"}}",
                f"An error was observed during this phase.\n"
                f"{jdump({'success': False, 'failure_reason': reason})}",
                fvs, fve, "json")
            add("success", ep,
                f"{q}\n\nRespond in a complete sentence.",
                f"No, this phase was not completed successfully. The failure was '{reason}'.",
                fvs, fve, "nl")
            add("success", ep,
                f"{q}\n\nAnswer only: Yes or No",
                "No", fvs, fve, "single")
        else:
            add("success", ep,
                f"{q}\n\nRespond with JSON: {{\"success\": <bool>, \"failure_reason\": \"<reason or null>\"}}",
                f"The phase was executed cleanly.\n"
                f"{jdump({'success': True, 'failure_reason': None})}",
                vs, ve, "json")
            add("success", ep,
                f"{q}\n\nRespond in a complete sentence.",
                f"Yes, this phase was completed successfully without errors.",
                vs, ve, "nl")
            add("success", ep,
                f"{q}\n\nAnswer only: Yes or No",
                "Yes", vs, ve, "single")

        # ── Failure reason (only for failed phases) ──────────────────
        if is_failed:
            reason = fail_map[pi]
            q = random.choice(FAILURE_QS)
            was_retried = pi in ep["retry_map"]
            fvs, fve = clip_range(ep, pi, MARGIN_BEFORE, MARGIN_AFTER, True)
            add("failure_reason", ep,
                f"{q}\nChoose from: {FAILURE_LIST}\n\n"
                f"Respond with JSON: {{\"phase\": \"<name>\", \"failure_reason\": \"<reason>\", \"was_retried\": <bool>}}",
                f"The instrument did not achieve the intended action.\n"
                f"{jdump({'phase': task, 'failure_reason': reason, 'was_retried': was_retried})}",
                fvs, fve, "json")
            add("failure_reason", ep,
                f"{q}\nChoose from: {FAILURE_LIST}\n\nRespond in a complete sentence.",
                f"The failure during '{task}' was '{reason}'. "
                f"{'It was retried immediately.' if was_retried else 'The procedure continued.'}",
                fvs, fve, "nl")

        # ── Completion detection ─────────────────────────────────────
        q = random.choice(COMPLETION_QS)
        # End-of-phase clip → Yes
        late_s = s2s(start + int(seg_len * 0.9))
        add("completion", ep,
            f"{q}\n\nAnswer only: Yes or No",
            "Yes", late_s, ve, "single")
        add("completion", ep,
            f"{q}\n\nRespond in a complete sentence.",
            f"Yes, the phase '{task}' has been completed.",
            late_s, ve, "nl")

        # ── Arm identification ───────────────────────────────────────
        q = random.choice(ARM_QS)
        add("arm_id", ep,
            f"{q}\nChoose from: Left arm, Right arm, Both arms\n\nAnswer only with your choice.",
            arm, vs, ve, "single")
        add("arm_id", ep,
            f"{q}\nChoose from: Left arm, Right arm, Both arms\n\nRespond in a complete sentence.",
            f"{arm} {'are' if arm == 'Both arms' else 'is'} performing '{task}'.",
            vs, ve, "nl")

        # ── Description (3 types, 1 phrasing each to control size) ───
        q = random.choice(DESC_QS)
        action_desc = task.split(": ", 1)[-1]
        add("description", ep,
            f"{q}\n\nRespond in 1-2 sentences.",
            f"{arm_of(task)} {'are' if arm_of(task)=='Both arms' else 'is'} {action_desc.lower()}. "
            f"This is phase {pi+1} of {n_phases} in the suturing procedure.",
            vs, ve, "nl")

    # ── Episode-level success ────────────────────────────────────────
    q = random.choice(EPISODE_QS)
    ep_vs = 0.0
    ep_ve = s2s(ep["length"])
    has_failures = len(fail_map) > 0

    if has_failures:
        fail_details = [{"phase": phases[si][2], "phase_index": si, "failure_reason": r}
                        for si, r in fail_map.items()]
        fail_desc = "; ".join(f"'{r}' during '{phases[si][2]}'" for si, r in fail_map.items())
        add("episode_success", ep,
            f"{q}\n\nRespond with JSON: {{\"success\": <bool>, \"failures\": [{{\"phase\": \"<name>\", \"phase_index\": <int>, \"failure_reason\": \"<reason>\"}}]}}",
            f"Multiple issues were observed.\n{jdump({'success': False, 'failures': fail_details})}",
            ep_vs, ep_ve, "json")
        add("episode_success", ep,
            f"{q}\n\nRespond in a complete sentence.",
            f"The suturing attempt had failures: {fail_desc}.",
            ep_vs, ep_ve, "nl")
    else:
        add("episode_success", ep,
            f"{q}\n\nRespond with JSON: {{\"success\": <bool>, \"failures\": []}}",
            f"The procedure was completed without errors.\n{jdump({'success': True, 'failures': []})}",
            ep_vs, ep_ve, "json")
        add("episode_success", ep,
            f"{q}\n\nRespond in a complete sentence.",
            f"Yes, this suturing attempt was completed successfully without any failures.",
            ep_vs, ep_ve, "nl")

    # ── Failure localization (for episodes with failures) ────────────
    if has_failures:
        # Pick first failure for localization
        first_fail_idx = min(fail_map.keys())
        fail_r = fail_map[first_fail_idx]
        fail_phase = phases[first_fail_idx][2]
        was_retried = first_fail_idx in ep["retry_map"]

        # Span ±1 phase around failure
        span_start_idx = max(0, first_fail_idx - 1)
        span_end_idx = min(n_phases - 1, first_fail_idx + (2 if was_retried else 1))
        span_vs = max(0.0, s2s(phases[span_start_idx][0]) - MARGIN_BEFORE)
        span_ve = min(s2s(ep["length"]), s2s(phases[span_end_idx][1]) + MARGIN_AFTER)

        q = "This clip spans multiple phases. A failure occurred. Identify the failed phase, failure type, and whether it was retried."
        add("failure_loc", ep,
            f"{q}\n\nRespond with JSON: {{\"failed_phase\": \"<name>\", \"failure_reason\": \"<reason>\", "
            f"\"was_retried\": <bool>, \"retry_successful\": <bool>}}",
            f"An error was detected in one of the phases.\n"
            f"{jdump({'failed_phase': fail_phase, 'failure_reason': fail_r, 'was_retried': was_retried, 'retry_successful': was_retried})}",
            span_vs, span_ve, "json")
        add("failure_loc", ep,
            f"{q}\n\nRespond in a complete sentence.",
            f"The failure occurred during '{fail_phase}': '{fail_r}'. "
            f"{'The phase was retried.' if was_retried else 'No retry was attempted.'}",
            span_vs, span_ve, "nl")

print(f"\nTotal samples: {len(samples)}")


# ── Validation split: 50 samples covering all task types ──────────────────────
print("\nCreating validation split...")

# Sample ~4 per task type to get ~50 total
val_samples = []
train_samples = []

# Group by task type
by_type = defaultdict(list)
for i, s in enumerate(samples):
    ttype = s["id"].rsplit("_", 1)[0]
    # Extract task type prefix
    for prefix in ["phase_id", "progress", "next_phase", "prev_phase", "ordering",
                    "success", "failure_reason", "episode_success", "completion",
                    "arm_id", "description", "failure_loc"]:
        if s["id"].startswith(prefix):
            by_type[prefix].append(i)
            break

n_types = len(by_type)
per_type = max(2, 50 // n_types)  # ~4 per type

val_indices = set()
for ttype, indices in by_type.items():
    # Sample from different formats
    fmt_groups = defaultdict(list)
    for idx in indices:
        fmt_groups[samples[idx]["answer_format"]].append(idx)

    chosen = []
    for fmt, fmt_indices in fmt_groups.items():
        n_from_fmt = max(1, per_type // len(fmt_groups))
        chosen.extend(random.sample(fmt_indices, min(n_from_fmt, len(fmt_indices))))

    # Trim to per_type
    if len(chosen) > per_type:
        chosen = random.sample(chosen, per_type)
    val_indices.update(chosen)

for i, s in enumerate(samples):
    if i in val_indices:
        val_samples.append(s)
    else:
        train_samples.append(s)

print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

# Val stats
val_types = Counter()
val_fmts = Counter()
for s in val_samples:
    for prefix in by_type:
        if s["id"].startswith(prefix):
            val_types[prefix] += 1
            break
    val_fmts[s["answer_format"]] += 1

print(f"  Val task coverage: {dict(val_types)}")
print(f"  Val format mix: {dict(val_fmts)}")


# ── Save ──────────────────────────────────────────────────────────────────────
train_path = os.path.join(OUT_DIR, "vrtb_suturing_train.json")
val_path = os.path.join(OUT_DIR, "vrtb_suturing_val.json")

with open(train_path, "w") as f:
    json.dump(train_samples, f, ensure_ascii=False)
print(f"\nSaved train → {train_path} ({os.path.getsize(train_path)/1e6:.0f} MB)")

with open(val_path, "w") as f:
    json.dump(val_samples, f, indent=2, ensure_ascii=False)
print(f"Saved val   → {val_path} ({os.path.getsize(val_path)/1e3:.0f} KB)")

# Final stats
fmt_counts = Counter(s["answer_format"] for s in train_samples)
print(f"\nTrain format distribution: {dict(fmt_counts)}")
print(f"Task distribution: {dict(task_type_counts)}")
