import json
import cv2
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
SUBJECT  = "Mar"
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / SUBJECT
OUT_DIR  = ROOT_DIR / "poc" / "overlays" / SUBJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Landmark index groups ──────────────────────────────────────────────────────
# Full 468-point face mesh tesselation edges (pairs)
# Using MediaPipe's canonical connection list (subset — all visible face edges)
FACE_CONNECTIONS = [
    # Silhouette
    10,338, 338,297, 297,332, 332,284, 284,251, 251,389, 389,356, 356,454,
    454,323, 323,361, 361,288, 288,397, 397,365, 365,379, 379,378, 378,400,
    400,377, 377,152, 152,148, 148,176, 176,149, 149,150, 150,136, 136,172,
    172,58,  58,132,  132,93,  93,234,  234,127, 127,162, 162,21,  21,54,
    54,103,  103,67,  67,109,  109,10,
    # Nose
    168,6, 6,197, 197,195, 195,5, 5,4, 4,1, 1,19, 19,94, 94,2,
    98,97, 97,2, 2,326, 326,327, 327,294,
    98,129, 129,49, 49,131, 131,134, 134,51, 51,5,
    294,358, 358,279, 279,360, 360,363, 363,281, 281,5,
    # Left eye
    33,7, 7,163, 163,144, 144,145, 145,153, 153,154, 154,155, 155,133,
    33,246, 246,161, 161,160, 160,159, 159,158, 158,157, 157,173, 173,133,
    # Right eye
    362,382, 382,381, 381,380, 380,374, 374,373, 373,390, 390,249, 249,263,
    362,398, 398,384, 384,385, 385,386, 386,387, 387,388, 388,466, 466,263,
    # Eyebrows
    46,53, 53,52, 52,65, 65,55,  70,63, 63,105, 105,66, 66,107,
    276,283, 283,282, 282,295, 295,285,  300,293, 293,334, 334,296, 296,336,
    # Lips outer
    61,146, 146,91, 91,181, 181,84, 84,17, 17,314, 314,405, 405,321,
    321,375, 375,291, 291,308, 308,324, 324,318, 318,402, 402,317, 317,14,
    14,87,  87,178, 178,88,  88,95,  95,61,
    # Lips inner
    78,95,  95,88,  88,178, 178,87,  87,14,  14,317, 317,402, 402,318,
    318,324, 324,308, 308,291, 291,375, 375,321, 321,405, 405,314, 314,17,
    17,84,  84,181, 181,91,  91,146, 146,61,  61,78,
]
# Build edge pairs
FACE_EDGES = [(FACE_CONNECTIONS[i], FACE_CONNECTIONS[i+1])
              for i in range(0, len(FACE_CONNECTIONS) - 1, 2)]

OUTER_LIP = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78]
INNER_LIP = [78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]

# Key landmarks to highlight with dots
KEY_LM = {
    "R_COMMISSURE":    61,
    "L_COMMISSURE":   291,
    "UPPER_LIP_TOP":    0,
    "UPPER_LIP_CTR":   13,
    "LOWER_LIP_CTR":   14,
    "NOSE_TIP":         1,
    "CHIN":           152,
}

# Colours (BGR)
C_MESH     = (0,  180,  0)    # dim green — face mesh edges
C_OUTER    = (0,  255, 255)   # cyan — outer lip
C_INNER    = (0,  120, 255)   # orange — inner lip
C_DOT_MESH = (0,  200,  0)    # mesh dots
C_DOT_KEY  = (0,   0, 255)    # red — key landmarks
C_DOT_LIP  = (255, 255,  0)   # yellow — lip key points


def draw_overlay(frame, lm, width, height):
    # 1. Face mesh edges — thin, dim
    for a, b in FACE_EDGES:
        if a >= len(lm) or b >= len(lm):
            continue
        p1 = (int(lm[a]['x'] * width), int(lm[a]['y'] * height))
        p2 = (int(lm[b]['x'] * width), int(lm[b]['y'] * height))
        cv2.line(frame, p1, p2, C_MESH, 1, cv2.LINE_AA)

    # 2. All 468 landmark dots — small
    for pt in lm:
        px = int(pt['x'] * width)
        py = int(pt['y'] * height)
        cv2.circle(frame, (px, py), 1, C_DOT_MESH, -1, cv2.LINE_AA)

    # 3. Outer lip contour — thick cyan
    pts_outer = np.array(
        [[int(lm[i]['x'] * width), int(lm[i]['y'] * height)]
         for i in OUTER_LIP if i < len(lm)], np.int32)
    if len(pts_outer) > 1:
        cv2.polylines(frame, [pts_outer], True, C_OUTER, 2, cv2.LINE_AA)

    # 4. Inner lip contour — thick orange
    pts_inner = np.array(
        [[int(lm[i]['x'] * width), int(lm[i]['y'] * height)]
         for i in INNER_LIP if i < len(lm)], np.int32)
    if len(pts_inner) > 1:
        cv2.polylines(frame, [pts_inner], True, C_INNER, 2, cv2.LINE_AA)

    # 5. Lip contour dots — bright yellow, medium size
    lip_set = set(OUTER_LIP + INNER_LIP)
    for i in lip_set:
        if i >= len(lm):
            continue
        px = int(lm[i]['x'] * width)
        py = int(lm[i]['y'] * height)
        cv2.circle(frame, (px, py), 3, C_DOT_LIP, -1, cv2.LINE_AA)

    # 6. Key landmarks — large red dot + label
    for name, idx in KEY_LM.items():
        if idx >= len(lm):
            continue
        px = int(lm[idx]['x'] * width)
        py = int(lm[idx]['y'] * height)
        cv2.circle(frame, (px, py), 5, C_DOT_KEY, -1, cv2.LINE_AA)
        cv2.putText(frame, name, (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_DOT_KEY, 1, cv2.LINE_AA)


def process_video(video_path: Path):
    landmarks_path = video_path.with_name(video_path.stem + "_landmarks.json")
    if not landmarks_path.exists():
        print(f"SKIP {video_path.name}: no landmarks file")
        return

    try:
        data = json.loads(landmarks_path.read_text())
        landmarks_frames = data.get('landmarks', [])
    except Exception as e:
        print(f"ERROR loading landmarks for {video_path.name}: {e}")
        return

    # VFR-safe FPS
    timestamps = []
    cap_ts = cv2.VideoCapture(str(video_path))
    if cap_ts.isOpened():
        while True:
            ret, _ = cap_ts.read()
            if not ret:
                break
            timestamps.append(cap_ts.get(cv2.CAP_PROP_POS_MSEC))
        cap_ts.release()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path.name}")
        return

    if len(timestamps) > 1:
        duration = (timestamps[-1] - timestamps[0]) / 1000.0
        fps = (len(timestamps) - 1) / duration if duration > 0 else 30
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUT_DIR / f"{video_path.stem}_overlay.mp4"
    out = cv2.VideoWriter(str(output_path),
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    print(f"Processing {video_path.name} → {output_path.name}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(landmarks_frames):
            draw_overlay(frame, landmarks_frames[frame_idx]['lm'], width, height)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done  {output_path.name}  ({frame_idx} frames)")


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist")
        return
    videos = sorted(DATA_DIR.glob("*.webm"))
    if not videos:
        print(f"No .webm files in {DATA_DIR}")
        return
    print(f"Found {len(videos)} videos for {SUBJECT}")
    for v in videos:
        process_video(v)


if __name__ == "__main__":
    main()
