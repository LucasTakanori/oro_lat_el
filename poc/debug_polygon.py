"""Render inner-lip polygon ROI + detected tip on a clip's peak frame.

Usage:  .venv/bin/python -m poc.debug_polygon
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from poc import scoring
from poc.build_dataset import pick_best_frame, _load_video_frames

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = Path("/tmp")

CLIPS = [
    ("Lucas", "latR_2026-04-14T22-45-45-045Z_s50"),
    ("Lucas", "latL_2026-04-14T22-46-02-097Z_s25"),
    ("Lucas", "latL_2026-04-14T17-03-01-783Z_s75"),
]


def render(subject: str, stem: str) -> None:
    sd = DATA_DIR / subject
    vid = sd / f"{stem}.webm"
    lmp = sd / f"{stem}_landmarks.json"
    if not (vid.exists() and lmp.exists()):
        print(f"SKIP missing {stem}")
        return
    task = stem.split("_")[0]
    lm_frames = json.loads(lmp.read_text())["landmarks"]
    frames = _load_video_frames(vid)
    if not frames or not lm_frames:
        return
    H, W = frames[0].shape[:2]
    rest_ref = scoring.resting_reference(lm_frames, H, W)

    best = pick_best_frame(frames, lm_frames, task)
    if best < 0:
        best = len(frames) - 1
    frame = frames[best].copy()
    lm = lm_frames[best]["lm"]
    ref_f = scoring.ref_from_landmarks(lm, H, W)
    ref_eff = scoring.RefGeometry(
        CL=rest_ref.CL, CR=rest_ref.CR,
        UL=ref_f.UL, LL=ref_f.LL, MC=rest_ref.MC,
        IC=rest_ref.IC, MH=float(np.linalg.norm(ref_f.UL - ref_f.LL)),
    )

    (x0, y0, x1, y1), roi_mask = scoring._mouth_mask_and_bbox(
        ref_eff, H, W, task=task, lm=lm)
    tongue_full, _, _ = scoring._hsv_tongue_mask(frame)
    tongue_inroi = (tongue_full & roi_mask).astype(bool)

    # overlay: ROI polygon outline (cyan), color mask inside ROI (yellow)
    vis = frame.copy()
    vis[tongue_inroi] = (0, 255, 255)
    # polygon outline
    poly = scoring._inner_lip_polygon(lm, H, W)
    if task.startswith("lat"):
        poly = scoring._extend_polygon_horizontally(poly, ref_eff, pad_frac=0.30)
    cv2.polylines(vis, [poly.astype(np.int32)], True, (255, 255, 0), 2)

    tip = scoring.detect_tongue_tip(frame, ref_eff, task, mirrored=False, lm=lm)
    if tip.xy is not None:
        cv2.circle(vis, tuple(tip.xy.astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(vis, f"conf={tip.conf:.2f} area={tip.blob_area_frac:.4f}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # lat score preview
    if task.startswith("lat") and tip.xy is not None:
        s = scoring.score_lateralization(tip.xy, ref_eff, task, mirrored=False)
        cv2.putText(vis, f"{task} score={s:.1f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # tight crop around mouth area
    pad = int(ref_eff.IC * 0.8)
    cx0 = max(0, x0 - pad); cy0 = max(0, y0 - pad)
    cx1 = min(W, x1 + pad); cy1 = min(H, y1 + pad)
    crop = vis[cy0:cy1, cx0:cx1]

    out_path = OUT / f"dbg_poly_{stem}.png"
    cv2.imwrite(str(out_path), crop)
    print(f"→ {out_path}  tip={None if tip.xy is None else tip.xy.tolist()} "
          f"area_frac={tip.blob_area_frac:.4f} conf={tip.conf:.2f}")


if __name__ == "__main__":
    for subj, stem in CLIPS:
        render(subj, stem)
