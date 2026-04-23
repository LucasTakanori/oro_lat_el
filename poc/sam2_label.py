"""Auto-label tongue masks per clip using SAM2 (ultralytics).

Per frame:
  - Positive click: mouth center (from MediaPipe inner-lip polygon centroid).
  - Negative clicks: two commissures (push mask away from lip tissue).

Output: poc/out/masks/<subject>_<stem>.npz  with keys:
    masks   bool array (n_frames, H, W)  — tongue mask per frame
    H, W    int                          — frame size
    prompts array of [cx, cy, CL_x, CL_y, CR_x, CR_y] per frame

Run:  .venv/bin/python -m poc.sam2_label              # all clips
      .venv/bin/python -m poc.sam2_label Lucas        # one subject
      .venv/bin/python -m poc.sam2_label Lucas latR_2026-04-14T22-45-45-045Z_s50
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

from poc.scoring import (
    LM,
    _hsv_tongue_mask,
    _inner_lip_polygon,
    _mouth_mask_and_bbox,
    _polygon_mask,
    _extend_polygon_horizontally,
    ref_from_landmarks,
    resting_reference,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "poc" / "out"
MASKS_DIR = OUT_DIR / "masks"
MASKS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sam2.1_t.pt"   # smallest SAM2 weights (~40 MB)


def _load_video(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


OUTER_UPPER_LIP = 0    # outer upper-lip center
OUTER_LOWER_LIP = 17   # outer lower-lip center
INNER_UPPER_LIP = 13   # inner upper-lip center (lip surface facing mouth)
INNER_LOWER_LIP = 14   # inner lower-lip center


def _prompt(lm, rest, frame_bgr: np.ndarray, task: str,
            H: int, W: int) -> tuple[list | None, list | None, list | None]:
    """Return (points, labels, bbox) for SAM2.

    Strategy (after fighting empty-mask cases on subjects with protruded tongue):
      - Bbox = aggressively lateral-extended inner-lip polygon bbox.
      - Positives:
        (1) HSV blob centroid inside polygon (best-guess tongue location).
        (2) Task-biased point near expected tip (helps when tongue protrudes
            far past commissure and blob centroid is inside mouth).
      - Negatives:
        outer upper/lower lip centers + commissures only.
        (Removed inner-lip 13/14 negatives — tongue passes through those
        landmarks when protruded, so including them actively suppressed
        the correct mask.)
    """
    ref = ref_from_landmarks(lm, H, W)
    poly = _inner_lip_polygon(lm, H, W)
    if task.startswith("lat"):
        poly = _extend_polygon_horizontally(poly, ref, pad_frac=0.60)
    roi_mask = _polygon_mask(poly, H, W)

    y_lo = float(ref.UL[1])
    y_hi = float(ref.LL[1])
    y_mid = 0.5 * (y_lo + y_hi)

    # HSV blob centroid inside the (wide) polygon — no strict-y constraint;
    # keep whatever reddish tissue is near/past the mouth opening.
    tongue_full, _, _ = _hsv_tongue_mask(frame_bgr)
    cand = tongue_full & roi_mask
    pos_primary = None
    if cand.sum() >= 100:
        num, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
            cand.astype(np.uint8), 8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            k = 1 + int(np.argmax(areas))
            pos_primary = [float(centroids[k, 0]), float(centroids[k, 1])]
    if pos_primary is None:
        pos_primary = [float(ref.MC[0]), y_mid]

    # Task-biased secondary positive: shift PAST the task-side commissure
    # toward where a protruded tongue tip would be. Slightly lowered (tongue
    # protrusions hang below lip line). For mild protrusion this sits in
    # the cavity (SAM absorbs it into the blob anyway); for strong protrusion
    # it lands on the external tongue tissue.
    mh = max(y_hi - y_lo, 10.0)
    drop = 0.4 * mh
    if task == "latR":
        target_x = float(rest.CL[0]) - 0.25 * float(rest.IC)  # past image-left commissure
        target_y = float(rest.CL[1]) + drop
    elif task == "latL":
        target_x = float(rest.CR[0]) + 0.25 * float(rest.IC)  # past image-right commissure
        target_y = float(rest.CR[1]) + drop
    else:  # elev — click high between lips
        target_x = float(ref.MC[0])
        target_y = y_lo - 0.2 * mh
    pos_secondary = [target_x, target_y]

    # Minimal negatives: outer upper + outer lower lip centers only.
    # Commissures are problematic — tongue wraps around them during strong
    # lateralization, and using them as negatives suppressed the correct mask.
    def p(i): return [float(lm[i]["x"]) * W, float(lm[i]["y"]) * H]
    neg = [
        p(OUTER_UPPER_LIP),
        p(OUTER_LOWER_LIP),
    ]
    points = [pos_primary, pos_secondary] + neg
    labels = [1, 1, 0, 0]

    # Bbox: don't use polygon bbox (collapses to a thin sliver when subject
    # holds tongue between nearly-closed lips — MediaPipe tracks lip-tissue
    # landmarks, not tongue). Instead use a generous rectangle around the
    # lower face: commissures ± 0.6·IC horizontally, nose bottom to chin
    # vertically. This always contains the tongue whether in-mouth or
    # protruded.
    nose_y = float(lm[LM["NOSE_TIP"]]["y"]) * H
    chin_y = float(lm[LM["CHIN"]]["y"]) * H
    if task.startswith("lat"):
        x_lo = min(rest.CL[0], rest.CR[0]) - 0.6 * rest.IC
        x_hi = max(rest.CL[0], rest.CR[0]) + 0.6 * rest.IC
    else:  # elev — narrower horizontally
        x_lo = min(rest.CL[0], rest.CR[0]) - 0.1 * rest.IC
        x_hi = max(rest.CL[0], rest.CR[0]) + 0.1 * rest.IC
    y_box_lo = nose_y + 0.3 * (chin_y - nose_y)
    y_box_hi = chin_y
    bbox = [int(max(0, x_lo)), int(max(0, y_box_lo)),
            int(min(W, x_hi)), int(min(H, y_box_hi))]
    return points, labels, bbox


def label_clip(model: SAM, subject: str, stem: str, force: bool = False) -> bool:
    subj_dir = DATA_DIR / subject
    vid = subj_dir / f"{stem}.webm"
    lmp = subj_dir / f"{stem}_landmarks.json"
    out_path = MASKS_DIR / f"{subject}_{stem}.npz"
    if out_path.exists() and not force:
        return True
    if not (vid.exists() and lmp.exists()):
        print(f"SKIP missing {subject}/{stem}")
        return False
    lm_frames = json.loads(lmp.read_text()).get("landmarks") or []
    frames = _load_video(vid)
    if not frames or not lm_frames:
        print(f"SKIP empty {subject}/{stem}")
        return False
    n = min(len(frames), len(lm_frames))
    H, W = frames[0].shape[:2]
    rest = resting_reference(lm_frames, H, W)

    task = stem.split("_")[0]
    masks = np.zeros((n, H, W), dtype=bool)
    prompts = np.zeros((n, 2), dtype=np.float32)   # positive click xy
    for i in range(n):
        pts, lbls, bbox = _prompt(lm_frames[i]["lm"], rest, frames[i],
                                   task, H, W)
        prompts[i] = pts[0]
        res = model(frames[i], points=[pts], labels=[lbls],
                    bboxes=[bbox], verbose=False)
        if res and res[0].masks is not None and len(res[0].masks.data) > 0:
            m = res[0].masks.data[0].cpu().numpy().astype(bool)
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[i] = m
    np.savez_compressed(out_path, masks=masks, H=H, W=W, prompts=prompts)
    print(f"OK   {subject}/{stem}  ({n} frames)  → {out_path.name}")
    return True


def main():
    args = sys.argv[1:]
    print(f"Loading SAM2 weights: {MODEL_NAME}")
    model = SAM(MODEL_NAME)
    try:
        import torch
        if torch.cuda.is_available():
            model.to("cuda")
            print("SAM2 on CUDA")
    except Exception:
        pass

    if len(args) == 2:
        label_clip(model, args[0], args[1])
        return
    if len(args) == 1:
        subjects = [args[0]]
    else:
        subjects = [p.name for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]

    for subj in subjects:
        sd = DATA_DIR / subj
        for meta in sorted(sd.glob("*_meta.json")):
            stem = meta.name.replace("_meta.json", "")
            label_clip(model, subj, stem)


if __name__ == "__main__":
    main()
