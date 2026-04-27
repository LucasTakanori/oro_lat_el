"""Render a validation video: predicted tip (red) vs ground-truth tip (green).

Runs the trained YOLOv8n-pose model on every frame of a clip, overlays
predicted tip + GT tip, writes an MP4.

Run:
    python -m poc.render_val_video                          # all Albert clips
    python -m poc.render_val_video --subject Albert --clip latR_2026-04-23T23-51-10-466Z_s100
    python -m poc.render_val_video --subject Albert --fold loso_Albert
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ANN_DIR  = ROOT / "poc" / "out" / "annotations"
RUNS_DIR = ROOT / "runs" / "tip"
OUT_DIR  = ROOT / "poc" / "out" / "val_videos"


def load_frames(subject: str, stem: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(DATA_DIR / subject / f"{stem}.webm"))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def render_clip(model, ann_path: Path, out_path: Path, conf_thresh: float = 0.25) -> None:
    ann     = json.loads(ann_path.read_text())
    subject = ann["subject"]
    stem    = ann["stem"]
    tips_gt = {int(k): v for k, v in ann.get("tips", {}).items()}
    no_tongue = {int(k) for k in ann.get("no_tongue", {}).keys()}

    frames = load_frames(subject, stem)
    if not frames:
        print(f"  SKIP {ann_path.name} — no frames")
        return

    H, W = frames[0].shape[:2]
    fps  = 15
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H),
    )

    errors = []
    for i, frame in enumerate(frames):
        vis = frame.copy()

        # ── ground truth ──────────────────────────────────────────────────────
        if i in tips_gt:
            gx, gy = int(tips_gt[i][0]), int(tips_gt[i][1])
            cv2.circle(vis, (gx, gy), 8, (0, 255, 0), -1)
            cv2.circle(vis, (gx, gy), 8, (0, 0, 0), 2)
        elif i in no_tongue:
            cv2.putText(vis, "NO TONGUE", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)

        # ── model prediction ──────────────────────────────────────────────────
        results = model(frame, verbose=False, conf=conf_thresh)
        pred_xy = None
        for r in results:
            if r.keypoints is None or len(r.keypoints.xy) == 0:
                continue
            kp      = r.keypoints.xy[0]    # (n_kpts, 2)
            kp_conf = r.keypoints.conf[0]  # (n_kpts,) visibility confidence
            tip_idx = 2 if len(kp) >= 3 else 0
            tip_vis_conf = float(kp_conf[tip_idx]) if kp_conf is not None else 1.0
            px, py = float(kp[tip_idx][0]), float(kp[tip_idx][1])
            # only draw tip if model predicts it as visible
            if tip_vis_conf >= 0.5 and (px > 0 or py > 0):
                pred_xy = (int(px), int(py))
            # draw commissures regardless (always visible)
            if len(kp) >= 3:
                for ki, color in [(0, (255, 165, 0)), (1, (255, 165, 0))]:
                    cx2, cy2 = float(kp[ki][0]), float(kp[ki][1])
                    if cx2 > 0 or cy2 > 0:
                        cv2.circle(vis, (int(cx2), int(cy2)), 5, color, -1)
            # show tip visibility confidence on frame
            cv2.putText(vis, f"vis:{tip_vis_conf:.2f}", (W - 120, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if tip_vis_conf >= 0.5 else (0, 0, 255), 2)

        if pred_xy is not None:
            cv2.circle(vis, pred_xy, 8, (0, 0, 255), -1)
            cv2.circle(vis, pred_xy, 8, (0, 0, 0), 2)
            # error in pixels if GT exists
            if i in tips_gt:
                gx, gy = tips_gt[i]
                err = np.hypot(pred_xy[0] - gx, pred_xy[1] - gy)
                errors.append(err)
                cv2.line(vis, (int(gx), int(gy)), pred_xy, (255, 255, 0), 1)

        # ── legend ────────────────────────────────────────────────────────────
        cv2.circle(vis, (20, H - 50), 8, (0, 255, 0), -1)
        cv2.putText(vis, "GT", (34, H - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(vis, (70, H - 50), 8, (0, 0, 255), -1)
        cv2.putText(vis, "Pred", (84, H - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f"f{i:03d}", (W - 80, H - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        writer.write(vis)

    writer.release()
    mae = float(np.mean(errors)) if errors else float("nan")
    print(f"  {out_path.name}  frames={len(frames)}  GT-frames={len(tips_gt)}  "
          f"MAE={mae:.1f}px  → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",  default="Albert")
    ap.add_argument("--fold",     default="loso_Albert")
    ap.add_argument("--out-name", default=None, help="Output subfolder name (default: same as fold)")
    ap.add_argument("--clip",     default=None, help="Specific stem to render (optional)")
    ap.add_argument("--conf",     type=float, default=0.25)
    args = ap.parse_args()

    weights = RUNS_DIR / args.fold / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights at {weights}")

    from ultralytics import YOLO
    model = YOLO(str(weights))
    print(f"Loaded {weights}")

    out_name = args.out_name or args.fold

    ann_files = sorted(ANN_DIR.glob(f"{args.subject}_*.json"))
    if args.clip:
        ann_files = [f for f in ann_files if args.clip in f.name]
    if not ann_files:
        print(f"No annotation files for subject={args.subject} clip={args.clip}")
        return

    for ann_path in ann_files:
        stem = ann_path.stem.replace(f"{args.subject}_", "", 1)
        out  = OUT_DIR / out_name / f"{args.subject}_{stem}.mp4"
        print(f"Rendering {ann_path.name}")
        render_clip(model, ann_path, out, conf_thresh=args.conf)

    print(f"\nVideos saved to {OUT_DIR / out_name}/")


if __name__ == "__main__":
    main()
