"""POC dataset builder.

Parses data/<subject>/*_meta.json, picks the peak tongue-extension frame via
asymmetric-redness × sharpness scoring (Option C), crops a task-adaptive ROI
(Option A), and writes:

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

from poc import scoring
from poc.visibility_clf import MODEL_PATH as VIS_MODEL_PATH, load_predictor

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "poc" / "out"
CROPS_DIR = OUT_DIR / "crops"
MASKS_DIR = OUT_DIR / "masks"
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


def _load_video_frames(video_path: Path) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _mouth_roi_wide(lm, H: int, W: int):
    """Wide symmetric ROI used for frame-scoring redness signal."""
    xs = [lm[j]["x"] * W for j in [LM["R_COMMISSURE"], LM["L_COMMISSURE"],
                                     LM["UPPER_LIP_CENTER"], LM["LOWER_LIP_CENTER"]]]
    ys = [lm[j]["y"] * H for j in [LM["R_COMMISSURE"], LM["L_COMMISSURE"],
                                     LM["UPPER_LIP_CENTER"], LM["LOWER_LIP_CENTER"]]]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    pad = int((x1 - x0) * 0.35)
    return max(0, x0 - pad), y0, min(W, x1 + pad), y1


def pick_best_frame(video_frames: list, landmark_frames: list, task: str,
                    skip_first: int = 20) -> int:
    """Option C — asymmetric redness × sharpness scorer.

    Returns the index of the frame with highest tongue-extension score.
    Falls back to the last valid frame if all scores are zero.
    """
    n = min(len(video_frames), len(landmark_frames))
    if n == 0:
        return -1
    best_score, best_idx = -1.0, n - 1

    for i in range(skip_first, n):
        frame = video_frames[i]
        lm = landmark_frames[i]["lm"]
        H, W = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()

        x0, y0, x1, y1 = _mouth_roi_wide(lm, H, W)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        # BGR → redness = 2R − G − B
        r = roi[:, :, 2].astype(float)
        g = roi[:, :, 1].astype(float)
        b = roi[:, :, 0].astype(float)
        redness = 2 * r - g - b
        mid_x = redness.shape[1] // 2
        mid_y = redness.shape[0] // 2

        if task == "latR":
            asym = redness[:, :mid_x].mean() - redness[:, mid_x:].mean()
        elif task == "latL":
            asym = redness[:, mid_x:].mean() - redness[:, :mid_x].mean()
        else:  # elev
            asym = redness[:mid_y, :].mean() - redness[mid_y:, :].mean()

        score = max(0.0, asym) * sharp
        if score > best_score:
            best_score, best_idx = score, i

    return best_idx


def crop_mouth(frame_bgr, lm, task: str):
    """Option A — task-adaptive asymmetric ROI.

    lat*  → wide landscape rectangle, center biased toward tongue side.
    elev  → taller portrait focused on oral cavity opening.
    """
    H, W = frame_bgr.shape[:2]
    rC = np.array([lm[LM["R_COMMISSURE"]]["x"] * W, lm[LM["R_COMMISSURE"]]["y"] * H])
    lC = np.array([lm[LM["L_COMMISSURE"]]["x"] * W, lm[LM["L_COMMISSURE"]]["y"] * H])
    uL = np.array([lm[LM["UPPER_LIP_CENTER"]]["x"] * W, lm[LM["UPPER_LIP_CENTER"]]["y"] * H])
    chin = np.array([lm[LM["CHIN"]]["x"] * W, lm[LM["CHIN"]]["y"] * H])

    ic = np.linalg.norm(rC - lC)  # inter-commissure = face-scale reference
    cx = (rC[0] + lC[0]) / 2
    cy = (rC[1] + lC[1]) / 2

    if task.startswith("lat"):
        half_w = ic * 1.4   # 2.8× ic total — catches extended tongue tip
        half_h = ic * 0.55
        if task == "latR":
            cx += ic * 0.25
        elif task == "latL":
            cx -= ic * 0.25
    else:  # elev
        half_w = ic * 0.65
        half_h = (chin[1] - uL[1]) * 0.65

    x0 = int(max(0, cx - half_w))
    x1 = int(min(W, cx + half_w))
    y0 = int(max(0, cy - half_h))
    y1 = int(min(H, cy + half_h))
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


def parse_clip(meta_path: Path, subject_dir: Path, visibility_fn=None):
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
    # cameraFlipped in the metadata refers to the UI selfie-mirror (CSS
    # `transform: scaleX(-1)`) for display only. The stored video + MediaPipe
    # landmarks are in RAW camera space where subject-right = image-left.
    # So we treat mirrored as False for scoring, regardless of the flag.
    mirrored = False

    video_frames = _load_video_frames(video_path)
    if not video_frames:
        return None

    best_idx = pick_best_frame(video_frames, lm_frames, task)
    if best_idx < 0:
        return None
    frame = video_frames[best_idx]
    sync_lm = lm_frames[best_idx]["lm"]

    crop = crop_mouth(frame, sync_lm, task)
    if crop is None:
        return None

    crop_name = f"{subject_dir.name}_{task}_{ts}_s{score}.png"
    cv2.imwrite(str(CROPS_DIR / crop_name), crop)

    geo = scoring.score_clip(video_frames, lm_frames, task,
                             mirrored=mirrored, visibility_fn=visibility_fn)

    mask_geo = {"lat_score": None, "elev_score": None,
                "n_valid": 0, "mean_tip_conf": 0.0, "mean_vis_prob": 0.0}
    mask_path = MASKS_DIR / f"{subject_dir.name}_{task}_{ts}_s{score}.npz"
    if mask_path.exists():
        d = np.load(mask_path)
        mask_geo = scoring.score_clip_with_masks(
            video_frames, lm_frames, d["masks"], task, mirrored=mirrored)

    row = {
        "subject": subject_dir.name,
        "task": task,
        "score": score,
        "crop_path": str((CROPS_DIR / crop_name).relative_to(ROOT)),
        "video_path": str(video_path.relative_to(ROOT)),
        "n_frames": len(lm_frames),
        "best_frame_idx": best_idx,
        "mirrored": int(mirrored),
        "lat_score": geo["lat_score"],
        "elev_score": geo["elev_score"],
        "geo_score": geo["lat_score"] if task.startswith("lat") else geo["elev_score"],
        "n_valid_frames": geo["n_valid"],
        "mean_tip_conf": geo["mean_tip_conf"],
        "mean_vis_prob": geo["mean_vis_prob"],
        "mask_lat_score": mask_geo["lat_score"],
        "mask_elev_score": mask_geo["elev_score"],
        "mask_score": mask_geo["lat_score"] if task.startswith("lat") else mask_geo["elev_score"],
        "mask_n_valid": mask_geo["n_valid"],
        "mask_mean_conf": mask_geo["mean_tip_conf"],
    }
    row.update(landmark_features(sync_lm))
    row.update(image_features(crop))
    return row


def main():
    visibility_fn = None
    if VIS_MODEL_PATH.exists():
        visibility_fn = load_predictor()
        print(f"[visibility] using trained model at {VIS_MODEL_PATH}")
    else:
        print("[visibility] no trained model — falling back to tip-confidence proxy")

    rows = []
    skipped = []
    for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for meta_path in sorted(subject_dir.glob("*_meta.json")):
            row = parse_clip(meta_path, subject_dir, visibility_fn=visibility_fn)
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
