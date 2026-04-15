"""POC dataset builder.

Parses data/<subject>/*_meta.json, synchronizes the video frames with the
MediaPipe face landmarks to find the 'Last Valid Frame' (highest quality hold),
crops the mouth ROI, and writes:

  poc/out/dataset.csv    one row per clip: landmark + crop features + label
  poc/out/crops/*.png    buccal-ROI crops (anonymized mouth region only)

Run:  python -m poc.build_dataset
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "poc" / "out"
CROPS_DIR = OUT_DIR / "crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

LM = {
    "R_COMMISSURE": 61,
    "L_COMMISSURE": 291,
    "UPPER_LIP_CENTER": 13,
    "LOWER_LIP_CENTER": 14,
    "LIP_LEFT_OUTER": 78,
    "LIP_RIGHT_OUTER": 308,
    "NOSE_TIP": 1,
    "CHIN": 152,
    "UPPER_LIP_TOP": 0,
}

META_RE = re.compile(r"^(latR|latL|elev)_(.+)_s(\d+)_meta\.json$")
ALLOWED_SCORES = {
    "latR": {0, 25, 50, 75, 100},
    "latL": {0, 25, 50, 75, 100},
    "elev": {0, 50, 100},
}


def extract_sync_data(video_path: Path, landmark_frames: list):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    
    video_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_frames.append(frame)
    cap.release()
    
    if not video_frames or not landmark_frames:
        return None, None
        
    # Synchronize: Use the last frame where we have both video and landmarks
    idx = min(len(video_frames), len(landmark_frames)) - 1
    if idx < 0:
        return None, None
        
    return video_frames[idx], landmark_frames[idx]["lm"]


def crop_mouth(frame_bgr, lm, pad_frac: float = 0.6):
    H, W = frame_bgr.shape[:2]
    idxs = [LM[k] for k in (
        "R_COMMISSURE", "L_COMMISSURE",
        "LIP_LEFT_OUTER", "LIP_RIGHT_OUTER",
        "UPPER_LIP_CENTER", "LOWER_LIP_CENTER", "UPPER_LIP_TOP",
    )]
    xs = [lm[i]["x"] * W for i in idxs]
    ys = [lm[i]["y"] * H for i in idxs]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    side = max(w, h) * (1 + 2 * pad_frac)
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(W, cx + side / 2))
    y1 = int(min(H, cy + side / 2))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return frame_bgr[y0:y1, x0:x1]


def landmark_features(lm):
    def p(i):
        return np.array([lm[i]["x"], lm[i]["y"], lm[i]["z"]], dtype=np.float64)

    rC = p(LM["R_COMMISSURE"])
    lC = p(LM["L_COMMISSURE"])
    uL = p(LM["UPPER_LIP_CENTER"])
    dL = p(LM["LOWER_LIP_CENTER"])
    nose = p(LM["NOSE_TIP"])
    chin = p(LM["CHIN"])
    inter_comm = np.linalg.norm(rC[:2] - lC[:2]) + 1e-9
    lip_gap = np.linalg.norm(uL[:2] - dL[:2])
    face_h = np.linalg.norm(nose[:2] - chin[:2]) + 1e-9
    mouth_center = (rC + lC) / 2
    return {
        "mouth_aspect": float(lip_gap / inter_comm),
        "mouth_open_norm": float(lip_gap / face_h),
        "inter_comm_norm": float(inter_comm / face_h),
        "nose_to_mouth_y": float((mouth_center[1] - nose[1]) / face_h),
        "rC_z": float(rC[2]),
        "lC_z": float(lC[2]),
        "lip_depth_diff": float(uL[2] - dL[2]),
    }


def image_features(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[..., 0] * 2.0  # OpenCV H is 0-180
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0
    tongue = (((H >= 330) & (H <= 360)) | ((H >= 0) & (H <= 20))) \
        & (S >= 0.25) & (V >= 0.35) & (V <= 0.9)
    return {
        "img_mean_H": float(H.mean()),
        "img_mean_S": float(S.mean()),
        "img_mean_V": float(V.mean()),
        "img_tongue_frac": float(tongue.mean()),
        "img_redness": float(((H < 20) | (H > 340)).mean()),
    }


def parse_clip(meta_path: Path, subject_dir: Path):
    m = META_RE.match(meta_path.name)
    if not m:
        return None
    task, ts, score_str = m.group(1), m.group(2), m.group(3)
    score = int(score_str)
    if score not in ALLOWED_SCORES[task]:
        return None
    video_path = subject_dir / f"{task}_{ts}_s{score}.webm"
    landmarks_path = subject_dir / f"{task}_{ts}_s{score}_landmarks.json"
    if not video_path.exists() or not landmarks_path.exists():
        return None
    try:
        lm_doc = json.loads(landmarks_path.read_text())
        lm_frames = lm_doc.get("landmarks") or []
        if not lm_frames:
            return None
    except Exception:
        return None
        
    frame, sync_lm = extract_sync_data(video_path, lm_frames)
    if frame is None or sync_lm is None:
        return None
        
    crop = crop_mouth(frame, sync_lm)
    if crop is None:
        return None
        
    crop_name = f"{subject_dir.name}_{task}_{ts}_s{score}.png"
    cv2.imwrite(str(CROPS_DIR / crop_name), crop)

    row = {
        "subject": subject_dir.name,
        "task": task,
        "score": score,
        "crop_path": str((CROPS_DIR / crop_name).relative_to(ROOT)),
        "video_path": str(video_path.relative_to(ROOT)),
        "n_frames": len(lm_frames),
    }
    row.update(landmark_features(sync_lm))
    row.update(image_features(crop))
    return row


def main():
    rows = []
    skipped = []
    for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for meta_path in sorted(subject_dir.glob("*_meta.json")):
            row = parse_clip(meta_path, subject_dir)
            tag = f"{subject_dir.name}/{meta_path.stem}"
            if row is None:
                skipped.append(tag)
                print(f"SKIP {tag}")
            else:
                rows.append(row)
                print(f"OK   {tag}")
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "dataset.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n=== Built {len(df)} samples  (skipped {len(skipped)}) ===")
    print(f"CSV: {out_csv}")
    print(f"Crops: {CROPS_DIR}")
    if not df.empty:
        print("\nSubjects:", sorted(df["subject"].unique()))
        print("\nSamples per task × score:")
        print(df.groupby(["task", "score"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
