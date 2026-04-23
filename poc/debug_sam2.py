"""Overlay SAM2 mask on peak frame for a clip. Render 4 frames from the clip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from poc.build_dataset import pick_best_frame, _load_video_frames

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MASKS_DIR = ROOT / "poc" / "out" / "masks"
OUT = Path("/tmp")


def show(subject: str, stem: str):
    sd = DATA_DIR / subject
    vid = sd / f"{stem}.webm"
    lmp = sd / f"{stem}_landmarks.json"
    npz = MASKS_DIR / f"{subject}_{stem}.npz"
    if not (vid.exists() and lmp.exists() and npz.exists()):
        print("missing"); return
    lm_frames = json.loads(lmp.read_text())["landmarks"]
    frames = _load_video_frames(vid)
    d = np.load(npz)
    masks = d["masks"]
    task = stem.split("_")[0]
    best = pick_best_frame(frames, lm_frames, task)
    n = min(len(frames), len(masks))
    idxs = [5, max(15, best - 10), best, min(n - 1, best + 5)]
    idxs = [max(0, min(n - 1, i)) for i in idxs]

    panels = []
    for i in idxs:
        f = frames[i].copy()
        m = masks[i]
        overlay = f.copy()
        overlay[m] = (0, 255, 255)
        vis = cv2.addWeighted(f, 0.55, overlay, 0.45, 0)
        if m.any():
            ys, xs = np.where(m)
            cv2.putText(vis, f"f{i}  n={m.sum()}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # task-direction extremum as candidate tip
            if task == "latR":
                k = xs.argmin(); col = (0, 0, 255)
            elif task == "latL":
                k = xs.argmax(); col = (0, 0, 255)
            else:
                k = ys.argmin(); col = (0, 0, 255)
            cv2.circle(vis, (int(xs[k]), int(ys[k])), 10, col, -1)
        else:
            cv2.putText(vis, f"f{i}  NO MASK", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        panels.append(vis)

    row = np.concatenate(panels, axis=1)
    # tight vertical crop
    h = row.shape[0]
    y0 = int(h * 0.35); y1 = int(h * 0.95)
    row = row[y0:y1]
    out = OUT / f"dbg_sam2_{stem}.png"
    cv2.imwrite(str(out), row)
    print(f"→ {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        show(args[0], args[1])
    else:
        show("Lucas", "latR_2026-04-14T22-45-45-045Z_s50")
