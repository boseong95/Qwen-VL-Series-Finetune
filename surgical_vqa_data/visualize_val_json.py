"""
Visualize JSON-format val samples with Q/A burned into video frames.

For each sample:
  1. Extract keyframes from the video clip
  2. Burn question + answer (with syntax-highlighted JSON) onto the frames
  3. Save as annotated video + single-frame card
"""

import json, os, re, subprocess, textwrap
import av
from PIL import Image, ImageDraw, ImageFont

VAL_PATH = "surgical_vqa_data/vrtb_suturing_val.json"
OUT_DIR = "surgical_vqa_data/val_visualizations_json"
os.makedirs(OUT_DIR, exist_ok=True)

with open(VAL_PATH) as f:
    samples = json.load(f)

json_samples = [s for s in samples if s.get("answer_format") == "json"]
print(f"Visualizing {len(json_samples)} JSON-format val samples...")

# ── Fonts ─────────────────────────────────────────────────────────────────────
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
    FONT_TITLE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
except OSError:
    FONT = FONT_BOLD = FONT_TITLE = FONT_SMALL = ImageFont.load_default()

# ── Colors ────────────────────────────────────────────────────────────────────
C_BG = (20, 20, 25)
C_PANEL = (35, 35, 42)
C_BORDER = (60, 60, 75)
C_TITLE = (100, 180, 255)
C_Q_LABEL = (130, 190, 255)
C_Q_TEXT = (210, 210, 220)
C_A_LABEL = (100, 255, 150)
C_COT = (180, 180, 150)  # chain-of-thought prefix
C_JSON_KEY = (100, 200, 255)
C_JSON_STR = (130, 255, 130)
C_JSON_BOOL = (255, 220, 80)
C_JSON_NULL = (255, 150, 80)
C_JSON_NUM = (255, 180, 100)
C_JSON_BRACE = (140, 140, 160)
C_FMT_BADGE = (70, 130, 250)


def draw_rounded_rect(draw, xy, fill, radius=6):
    x0, y0, x1, y1 = xy
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.pieslice([x0, y0, x0 + 2*radius, y0 + 2*radius], 180, 270, fill=fill)
    draw.pieslice([x1 - 2*radius, y0, x1, y0 + 2*radius], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - 2*radius, x0 + 2*radius, y1], 90, 180, fill=fill)
    draw.pieslice([x1 - 2*radius, y1 - 2*radius, x1, y1], 0, 90, fill=fill)


def draw_json_lines(draw, x, y, json_str, font, max_lines=15):
    """Render pretty-printed JSON with syntax coloring. Returns final y."""
    try:
        parsed = json.loads(json_str)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        pretty = json_str

    lines = pretty.split("\n")[:max_lines]
    for jline in lines:
        cx = x
        # Try to parse "key": value
        kv = re.match(r'^(\s*)"(.+?)":\s*(.+?),?\s*$', jline)
        if kv:
            indent_str, key, val = kv.groups()
            val_clean = val.rstrip(",")
            has_comma = val.endswith(",")

            draw.text((cx, y), indent_str + '"', fill=C_JSON_BRACE, font=font)
            cx += draw.textlength(indent_str + '"', font=font)
            draw.text((cx, y), key, fill=C_JSON_KEY, font=font)
            cx += draw.textlength(key, font=font)
            draw.text((cx, y), '": ', fill=C_JSON_BRACE, font=font)
            cx += draw.textlength('": ', font=font)

            if val_clean in ("true", "false"):
                vc = C_JSON_BOOL
            elif val_clean == "null":
                vc = C_JSON_NULL
            elif val_clean.startswith('"'):
                vc = C_JSON_STR
            elif val_clean.startswith("[") or val_clean.startswith("{"):
                vc = C_JSON_BRACE
            else:
                vc = C_JSON_NUM

            draw.text((cx, y), val_clean, fill=vc, font=font)
            if has_comma:
                cx += draw.textlength(val_clean, font=font)
                draw.text((cx, y), ",", fill=C_JSON_BRACE, font=font)
        else:
            draw.text((cx, y), jline, fill=C_JSON_BRACE, font=font)

        y += 17
    return y


def make_annotated_frame(frame_img, sample_id, question, answer, answer_format):
    """Create a wide annotated frame: video on left, Q/A panel on right."""
    fw, fh = frame_img.size

    # Panel width
    panel_w = 580
    card_w = fw + panel_w
    card_h = max(fh, 520)

    card = Image.new("RGB", (card_w, card_h), C_BG)
    # Paste video frame on the left
    card.paste(frame_img, (0, (card_h - fh) // 2))

    draw = ImageDraw.Draw(card)

    # Draw panel background
    px = fw + 5
    draw_rounded_rect(draw, (px, 5, card_w - 5, card_h - 5), C_PANEL, radius=8)

    x = px + 12
    y = 15

    # Badge + ID
    draw_rounded_rect(draw, (x, y, x + 50, y + 20), C_FMT_BADGE, radius=4)
    draw.text((x + 6, y + 2), "JSON", fill=(255, 255, 255), font=FONT_SMALL)
    draw.text((x + 58, y + 2), sample_id, fill=(160, 160, 170), font=FONT_SMALL)
    y += 30

    # Separator
    draw.line([(x, y), (card_w - 17, y)], fill=C_BORDER, width=1)
    y += 8

    # Question
    draw.text((x, y), "QUESTION", fill=C_Q_LABEL, font=FONT_TITLE)
    y += 22
    q_clean = question.replace("<video>\n", "").replace("<image>\n", "")
    # Truncate the phase list if present
    if "Choose from:" in q_clean:
        parts = q_clean.split("Choose from:")
        q_display = parts[0].strip() + "\n[18 phase choices...]"
        # Check for JSON schema instruction after the choices
        for suffix in ["Respond with JSON:", "Respond in"]:
            if suffix in q_clean:
                q_display += "\n" + q_clean.split(suffix)[-1].strip()
                q_display = parts[0].strip() + "\n[18 phase choices...]\n" + suffix + " " + q_clean.split(suffix)[-1].strip()
                break
    elif "Choices:" in q_clean:
        parts = q_clean.split("Choices:")
        q_display = parts[0].strip() + "\n[18 phase choices...]"
        for suffix in ["Respond with JSON:"]:
            if suffix in q_clean:
                q_display = parts[0].strip() + "\n[18 phase choices...]\n" + suffix + " " + q_clean.split(suffix)[-1].strip()
                break
    else:
        q_display = q_clean

    for line in q_display.split("\n"):
        for wline in textwrap.wrap(line, width=62) or [""]:
            draw.text((x + 2, y), wline, fill=C_Q_TEXT, font=FONT_SMALL)
            y += 15
    y += 8

    # Separator
    draw.line([(x, y), (card_w - 17, y)], fill=C_BORDER, width=1)
    y += 8

    # Answer
    draw.text((x, y), "GROUND TRUTH", fill=C_A_LABEL, font=FONT_TITLE)
    y += 22

    # Split answer into CoT prefix and JSON
    json_match = re.search(r'(\{.*\}|\[.*\])', answer, re.DOTALL)
    if json_match:
        cot = answer[:json_match.start()].strip()
        json_str = json_match.group(0)

        # CoT reasoning prefix
        if cot:
            for wline in textwrap.wrap(cot, width=62)[:3]:
                draw.text((x + 2, y), wline, fill=C_COT, font=FONT_SMALL)
                y += 15
            y += 5

        # JSON with syntax highlighting
        y = draw_json_lines(draw, x + 2, y, json_str, FONT, max_lines=14)
    else:
        for wline in textwrap.wrap(answer, width=62)[:8]:
            draw.text((x + 2, y), wline, fill=(220, 220, 220), font=FONT)
            y += 17

    return card


def extract_frames_pyav(video_path, start_sec, end_sec, target_fps=5):
    """Extract frames from a clip at target_fps."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate) if stream.average_rate else 20.0

    if start_sec > 0:
        container.seek(int(start_sec * av.time_base), any_frame=False)

    all_frames = []
    for frame in container.decode(video=0):
        t = float(frame.pts * stream.time_base) if frame.pts is not None else 0
        if t < start_sec:
            continue
        if t > end_sec:
            break
        all_frames.append((t, frame.to_image()))

    container.close()

    if not all_frames:
        return []

    # Subsample to target_fps
    clip_dur = end_sec - start_sec
    n_target = max(4, int(clip_dur * target_fps))
    if len(all_frames) > n_target:
        indices = [int(i * len(all_frames) / n_target) for i in range(n_target)]
        all_frames = [all_frames[i] for i in indices]

    return all_frames


# ── Process each JSON val sample ──────────────────────────────────────────────
for i, s in enumerate(json_samples):
    sid = s["id"]
    video_path = s.get("video", "")
    vs = s.get("video_start", 0)
    ve = s.get("video_end", 10)
    question = s["conversations"][0]["value"]
    answer = s["conversations"][1]["value"]
    fmt = s.get("answer_format", "json")

    if not os.path.exists(video_path):
        print(f"  SKIP {sid}: video not found")
        continue

    # Extract frames at 1fps (matching --fps 1 in train_surgical_vqa.sh)
    TARGET_FPS = 1
    frames = extract_frames_pyav(video_path, vs, ve, target_fps=TARGET_FPS)
    if not frames:
        print(f"  SKIP {sid}: no frames decoded")
        continue

    # Create annotated card from middle frame
    mid_idx = len(frames) // 2
    _, mid_frame = frames[mid_idx]
    card = make_annotated_frame(mid_frame, sid, question, answer, fmt)
    card_path = os.path.join(OUT_DIR, f"{sid}_card.png")
    card.save(card_path)

    # Create annotated video (burn Q/A into each frame)
    video_out_path = os.path.join(OUT_DIR, f"{sid}.mp4")
    # Write frames as annotated images, then encode
    frame_dir = os.path.join(OUT_DIR, f"_tmp_{sid}")
    os.makedirs(frame_dir, exist_ok=True)
    for fi, (t, frame) in enumerate(frames):
        annotated = make_annotated_frame(frame, sid, question, answer, fmt)
        annotated.save(os.path.join(frame_dir, f"{fi:04d}.png"))

    # Encode to video at same fps
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(TARGET_FPS), "-i", os.path.join(frame_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-crf", "20", "-loglevel", "error", video_out_path,
    ], check=True)

    # Clean up temp frames
    for f_file in os.listdir(frame_dir):
        os.remove(os.path.join(frame_dir, f_file))
    os.rmdir(frame_dir)

    print(f"  [{i+1}/{len(json_samples)}] {sid}")

print(f"\nDone. Output → {OUT_DIR}/")
print(f"  {len(json_samples)} cards (.png) + {len(json_samples)} annotated videos (.mp4)")
