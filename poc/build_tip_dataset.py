"""Build YOLO-pose tip dataset from annotator JSON files.

Per annotation JSON → per annotated frame (visible or NO_TONGUE):
  - Full frame with chosen preprocessing saved as JPEG
  - YOLO-pose label: 0  cx cy w h  CL_x CL_y 2  CR_x CR_y 2  tip_x tip_y vis
      3 keypoints: left commissure, right commissure, tongue tip
      CL/CR always visible (vis=2), tip vis=2 if annotated, vis=0 if no_tongue
  - Bbox from MediaPipe inner-lip polygon; lat tasks extended ±0.6·IC

Pipelines:
  clahe             — CLAHE only (original)
  clahe_unsharp     — CLAHE then unsharp mask
  gamma_clahe_unsharp — gamma→CLAHE→unsharp

Output (example for clahe_unsharp):
  dataset_clahe_unsharp/images/<subject>_<stem>_<f>.jpg
  dataset_clahe_unsharp/labels/<subject>_<stem>_<f>.txt
  dataset_clahe_unsharp/tip_loso_<val_subject>.yaml

Run:
  python -m poc.build_tip_dataset                         # clahe (default)
  python -m poc.build_tip_dataset --pipeline clahe_unsharp
  python -m poc.build_tip_dataset --pipeline gamma_clahe_unsharp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

from poc.scoring import (
    _extend_polygon_horizontally,
    _inner_lip_polygon,
    ref_from_landmarks,
)

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
ANNOT_DIR = ROOT / "poc" / "out" / "annotations"

LM_R_COMMISSURE = 61
LM_L_COMMISSURE = 291


# ── preprocessing functions ───────────────────────────────────────────────────

def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    merged = cv2.merge([clahe.apply(L), a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _apply_gamma(bgr: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    lut = (np.arange(256) / 255.0) ** gamma * 255
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def _apply_unsharp(bgr: np.ndarray, strength: float = 1.5, ksize: int = 5) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (ksize, ksize), 0)
    return cv2.addWeighted(bgr, 1 + strength, blurred, -strength, 0)


def _pipeline_clahe(bgr: np.ndarray) -> np.ndarray:
    return _apply_clahe(bgr)


def _pipeline_clahe_unsharp(bgr: np.ndarray) -> np.ndarray:
    return _apply_unsharp(_apply_clahe(bgr))


def _pipeline_gamma_clahe_unsharp(bgr: np.ndarray) -> np.ndarray:
    return _apply_unsharp(_apply_clahe(_apply_gamma(bgr, gamma=0.5)))


PIPELINES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "clahe":               _pipeline_clahe,
    "clahe_unsharp":       _pipeline_clahe_unsharp,
    "gamma_clahe_unsharp": _pipeline_gamma_clahe_unsharp,
}


# ── dataset helpers ───────────────────────────────────────────────────────────

def _commissure_kpts(lm: list, H: int, W: int) -> tuple[tuple, tuple]:
    cl = lm[LM_R_COMMISSURE]
    cr = lm[LM_L_COMMISSURE]
    # clamp to [0,1] — lat extension can push landmark slightly past image edge
    clamp = lambda v: min(max(float(v), 0.0), 1.0)
    return (clamp(cl["x"]), clamp(cl["y"]), 2), (clamp(cr["x"]), clamp(cr["y"]), 2)


def _load_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def _lip_bbox(lm: list, H: int, W: int, task: str) -> Optional[tuple[float, float, float, float]]:
    ref = ref_from_landmarks(lm, H, W)
    poly = _inner_lip_polygon(lm, H, W)

    if task.startswith("lat"):
        poly = _extend_polygon_horizontally(poly, ref, pad_frac=0.60)
    elif task == "elev":
        # IC-based upward expansion — stable even when mouth barely open (MH≈0)
        up_pad = 0.50 * max(ref.IC, 1.0)
        poly = poly.copy()
        poly[poly[:, 1] < ref.MC[1], 1] -= up_pad

    x0 = float(np.clip(poly[:, 0].min(), 0, W))
    x1 = float(np.clip(poly[:, 0].max(), 0, W))
    y0 = float(np.clip(poly[:, 1].min(), 0, H))
    y1 = float(np.clip(poly[:, 1].max(), 0, H))

    # Enforce minimum height ≥ 20% of IC so barely-open-mouth bbox isn't a sliver
    min_h = 0.20 * max(ref.IC, 1.0)
    if y1 - y0 < min_h:
        mc_y = ref.MC[1]
        y0 = max(0.0, min(y0, mc_y - min_h / 2))
        y1 = min(float(H), max(y1, mc_y + min_h / 2))

    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return None

    cx = (x0 + x1) / 2.0 / W
    cy = (y0 + y1) / 2.0 / H
    return cx, cy, bw / W, bh / H


def _write_label(path: Path, bbox: tuple,
                 cl: tuple, cr: tuple,
                 tip_xy: Optional[list], vis: int, W: int, H: int) -> None:
    cx, cy, bw, bh = bbox
    cl_x, cl_y, cl_v = cl
    cr_x, cr_y, cr_v = cr
    if vis == 2 and tip_xy is not None:
        tx, ty = tip_xy[0] / W, tip_xy[1] / H
    else:
        tx, ty, vis = 0.0, 0.0, 0
    path.write_text(
        f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
        f"{cl_x:.6f} {cl_y:.6f} {cl_v} "
        f"{cr_x:.6f} {cr_y:.6f} {cr_v} "
        f"{tx:.6f} {ty:.6f} {vis}\n"
    )


def process_annotation(annot_path: Path,
                        img_dir: Path,
                        lbl_dir: Path,
                        preprocess: Callable[[np.ndarray], np.ndarray]) -> list[str]:
    """Process one annotation JSON. Returns list of written image stems."""
    ann = json.loads(annot_path.read_text())
    subject   = ann["subject"]
    stem      = ann["stem"]
    task      = ann["task"]
    tips      = {int(k): v for k, v in ann.get("tips", {}).items()}
    no_tongue = {int(k): True for k in ann.get("no_tongue", {}).keys()}

    annotated = set(tips) | set(no_tongue)
    if not annotated:
        print(f"  SKIP {annot_path.name} — no annotations")
        return []

    video_path = DATA_DIR / subject / f"{stem}.webm"
    lm_path    = DATA_DIR / subject / f"{stem}_landmarks.json"

    if not video_path.exists():
        print(f"  SKIP {annot_path.name} — video missing: {video_path}")
        return []
    if not lm_path.exists():
        print(f"  SKIP {annot_path.name} — landmarks missing: {lm_path}")
        return []

    frames    = _load_video(video_path)
    lm_doc    = json.loads(lm_path.read_text())
    lm_frames = lm_doc.get("landmarks", [])

    written = []
    for f_idx in sorted(annotated):
        if f_idx >= len(frames) or f_idx >= len(lm_frames):
            continue

        frame = frames[f_idx]
        H, W  = frame.shape[:2]
        lm    = lm_frames[f_idx]["lm"]

        bbox = _lip_bbox(lm, H, W, task)
        if bbox is None:
            continue

        cl, cr = _commissure_kpts(lm, H, W)

        if f_idx in tips:
            vis    = 2
            tip_xy = tips[f_idx]
        else:
            vis    = 0
            tip_xy = None

        file_stem = f"{subject}_{stem}_{f_idx:04d}"
        img_path  = img_dir / f"{file_stem}.jpg"
        lbl_path  = lbl_dir / f"{file_stem}.txt"

        cv2.imwrite(str(img_path), preprocess(frame), [cv2.IMWRITE_JPEG_QUALITY, 95])
        _write_label(lbl_path, bbox, cl, cr, tip_xy, vis, W, H)
        written.append(file_stem)

    return written


def _write_yaml(val_subject: str, all_stems: dict[str, list[str]],
                dataset_dir: Path, img_dir: Path) -> Path:
    train_imgs, val_imgs = [], []
    for subj, stems in all_stems.items():
        paths = [str(img_dir / f"{s}.jpg") for s in stems]
        if subj == val_subject:
            val_imgs.extend(paths)
        else:
            train_imgs.extend(paths)

    train_txt = dataset_dir / f"train_{val_subject}.txt"
    val_txt   = dataset_dir / f"val_{val_subject}.txt"
    train_txt.write_text("\n".join(train_imgs) + "\n")
    val_txt.write_text("\n".join(val_imgs) + "\n")

    doc = {
        "path":      str(dataset_dir),
        "train":     str(train_txt),
        "val":       str(val_txt),
        "kpt_shape": [3, 3],
        "nc":        1,
        "names":     ["tongue"],
    }
    out = dataset_dir / f"tip_loso_{val_subject}.yaml"
    out.write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="clahe", choices=list(PIPELINES),
                    help="Preprocessing pipeline (default: clahe)")
    args = ap.parse_args()

    pipeline_name = args.pipeline
    preprocess    = PIPELINES[pipeline_name]

    # dataset_clahe stays at "dataset/" for backward compat;
    # others get their own dir
    if pipeline_name == "clahe":
        dataset_dir = ROOT / "dataset"
    else:
        dataset_dir = ROOT / f"dataset_{pipeline_name}"

    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline : {pipeline_name}")
    print(f"Output   : {dataset_dir}")

    annot_files = sorted(ANNOT_DIR.glob("*.json"))
    if not annot_files:
        print(f"No annotation files found in {ANNOT_DIR}")
        return

    stems_by_subject: dict[str, list[str]] = {}
    total = 0

    for ap_path in annot_files:
        ann     = json.loads(ap_path.read_text())
        subject = ann["subject"]
        print(f"[{subject}] {ap_path.name}")
        written = process_annotation(ap_path, img_dir, lbl_dir, preprocess)
        stems_by_subject.setdefault(subject, []).extend(written)
        total += len(written)
        print(f"  → {len(written)} frames")

    print(f"\nTotal frames written: {total}")
    print(f"Subjects: {sorted(stems_by_subject)}")

    for subj in sorted(stems_by_subject):
        yp = _write_yaml(subj, stems_by_subject, dataset_dir, img_dir)
        n_val   = len(stems_by_subject[subj])
        n_train = total - n_val
        print(f"  YAML (val={subj}): {yp.name}  train={n_train}  val={n_val}")


if __name__ == "__main__":
    main()
