"""Visibility classifier: P(tongue tip visible in frame).

Small logistic regression on hand-crafted features. Needed to distinguish
elevation=100 (tip hidden behind teeth, mouth open) from elevation=0
(no movement, tip not protruded). HSV alone can't do this because it
always returns the largest red blob — it has no negative class.

Features (per frame):
    blob_area_frac   HSV-tongue blob area / mouth ROI area
    sat_mean         mean saturation of blob
    hue_concentration fraction of blob in tight tongue-hue band
    mouth_open       mouth-aspect ratio (MH / IC)
    blob_y_norm      blob centroid y within mouth bbox (None → 0)
    blob_below_upper 1.0 if blob y > upper-lip y, else 0.0

Pseudo-labels (unsupervised):
    POSITIVE = HSV detection with conf > 0.6 AND inside mouth
    NEGATIVE = first 5 frames of each clip (resting, pre-movement)
               OR frames with blob_area_frac < 0.002
               OR frames with mouth_open < 0.1 (mouth closed)

Usage:
    python -m poc.visibility_clf train   # builds model, saves pickle
    python -m poc.visibility_clf eval    # LOSO eval, prints metrics
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from poc.scoring import (
    LM,
    REST_FRAMES,
    _hsv_tongue_mask,
    _mouth_mask_and_bbox,
    mouth_open_prob,
    ref_from_landmarks,
    resting_reference,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "poc" / "out"
MODEL_PATH = OUT_DIR / "visibility_clf.pkl"
FEATURES_PATH = OUT_DIR / "visibility_features.csv"

FEAT_COLS = ["blob_area_frac", "sat_mean", "hue_concentration",
             "mouth_open", "blob_y_norm", "blob_below_upper"]


def extract_frame_features(frame_bgr: np.ndarray, ref, task: str = "elev",
                            lm=None) -> dict:
    """Compute visibility features for one frame given ref geometry."""
    H, W = frame_bgr.shape[:2]
    (x0, y0, x1, y1), mouth_full = _mouth_mask_and_bbox(ref, H, W, task=task, lm=lm)
    tongue_full, S, _V = _hsv_tongue_mask(frame_bgr)
    tongue = tongue_full & mouth_full

    mouth_area = float((x1 - x0) * (y1 - y0)) + 1e-9
    if tongue.sum() == 0:
        return dict(blob_area_frac=0.0, sat_mean=0.0, hue_concentration=0.0,
                    mouth_open=mouth_open_prob(ref),
                    blob_y_norm=0.0, blob_below_upper=0.0)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(tongue, 8)
    if num <= 1:
        return dict(blob_area_frac=0.0, sat_mean=0.0, hue_concentration=0.0,
                    mouth_open=mouth_open_prob(ref),
                    blob_y_norm=0.0, blob_below_upper=0.0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    blob = (labels == best)

    blob_area_frac = float(blob.sum()) / mouth_area
    sat_mean = float(S[blob].mean())

    # hue concentration: fraction of blob in [345°, 360°] ∪ [0°, 10°]
    # (tight tongue-pink band) vs the wider HSV range we used to segment.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    Hc = hsv[..., 0] * 2.0
    tight = ((Hc >= 345) | (Hc <= 10)) & blob
    hue_concentration = float(tight.sum()) / float(blob.sum() + 1e-9)

    ys, xs = np.where(blob)
    cy = float(ys.mean())
    cx = float(xs.mean())
    mh = max(y1 - y0, 1)
    blob_y_norm = (cy - y0) / mh
    blob_below_upper = 1.0 if cy > ref.UL[1] else 0.0

    return dict(
        blob_area_frac=blob_area_frac,
        sat_mean=sat_mean,
        hue_concentration=hue_concentration,
        mouth_open=mouth_open_prob(ref),
        blob_y_norm=float(blob_y_norm),
        blob_below_upper=blob_below_upper,
    )


def _pseudo_label(feat: dict, is_rest: bool) -> int | None:
    """Return 1 (visible), 0 (not visible), or None (ambiguous — skip)."""
    if is_rest:
        return 0
    if feat["blob_area_frac"] < 0.002:
        return 0
    if feat["mouth_open"] < 0.10:
        return 0
    # Strong positive: large blob, high sat, concentrated in tongue hue
    if (feat["blob_area_frac"] > 0.03 and feat["sat_mean"] > 0.45
            and feat["hue_concentration"] > 0.55):
        return 1
    return None


def _load_video(path: Path) -> list:
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


def build_feature_table() -> pd.DataFrame:
    """Iterate all clips → extract per-frame features + pseudo-labels."""
    rows = []
    for subj_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for meta in sorted(subj_dir.glob("*_meta.json")):
            stem = meta.name.replace("_meta.json", "")
            vid = subj_dir / f"{stem}.webm"
            lmp = subj_dir / f"{stem}_landmarks.json"
            if not (vid.exists() and lmp.exists()):
                continue
            lm_frames = json.loads(lmp.read_text()).get("landmarks") or []
            frames = _load_video(vid)
            if not frames or not lm_frames:
                continue
            H, W = frames[0].shape[:2]
            rest = resting_reference(lm_frames, H, W)
            task = stem.split("_")[0]
            for i, fr in enumerate(frames[: len(lm_frames)]):
                lm = lm_frames[i]["lm"]
                ref_f = ref_from_landmarks(lm, H, W)
                # Same effective-ref trick as scoring.py.
                from poc.scoring import RefGeometry
                ref = RefGeometry(
                    CL=rest.CL, CR=rest.CR,
                    UL=ref_f.UL, LL=ref_f.LL, MC=rest.MC,
                    IC=rest.IC, MH=float(np.linalg.norm(ref_f.UL - ref_f.LL)),
                )
                feat = extract_frame_features(fr, ref, task=task, lm=lm)
                lbl = _pseudo_label(feat, is_rest=(i < REST_FRAMES))
                feat.update(subject=subj_dir.name, clip=stem, task=task,
                            frame_idx=i, label=lbl)
                rows.append(feat)
            print(f"OK   {subj_dir.name}/{stem}  ({len(frames)} frames)")
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"→ {FEATURES_PATH}   ({len(df)} rows, "
          f"pos={int((df.label==1).sum())} neg={int((df.label==0).sum())} "
          f"ambiguous={int(df.label.isna().sum())})")
    return df


def _loso_auc(df_lab: pd.DataFrame) -> float:
    X = df_lab[FEAT_COLS].values
    y = df_lab["label"].values.astype(int)
    groups = df_lab["subject"].values
    logo = LeaveOneGroupOut()
    probs = np.full_like(y, np.nan, dtype=float)
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            probs[te] = float(y[tr].mean())
            continue
        model = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(max_iter=1000, C=1.0))])
        model.fit(X[tr], y[tr])
        probs[te] = model.predict_proba(X[te])[:, 1]
    try:
        return float(roc_auc_score(y, probs))
    except ValueError:
        return float("nan")


def train(df: pd.DataFrame | None = None) -> Pipeline:
    if df is None:
        df = pd.read_csv(FEATURES_PATH) if FEATURES_PATH.exists() else build_feature_table()
    df_lab = df.dropna(subset=["label"]).copy()
    df_lab["label"] = df_lab["label"].astype(int)
    print(f"\nTrain set: {len(df_lab)}  pos={int((df_lab.label==1).sum())}  "
          f"neg={int((df_lab.label==0).sum())}")
    auc = _loso_auc(df_lab)
    print(f"LOSO ROC-AUC = {auc:.3f}")

    # Final model fit on ALL labeled data (used at inference).
    X = df_lab[FEAT_COLS].values
    y = df_lab["label"].values.astype(int)
    model = Pipeline([("sc", StandardScaler()),
                      ("m", LogisticRegression(max_iter=1000, C=1.0))])
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feat_cols": FEAT_COLS, "loso_auc": auc}, f)
    print(f"→ {MODEL_PATH}")
    return model


def load_predictor():
    """Return a callable(frame_bgr, ref, tip) -> float in [0,1]."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} missing — run `python -m poc.visibility_clf train` first")
    with open(MODEL_PATH, "rb") as f:
        blob = pickle.load(f)
    model, feat_cols = blob["model"], blob["feat_cols"]

    def predict(frame_bgr, ref, _tip=None, task: str = "elev", lm=None) -> float:
        feat = extract_frame_features(frame_bgr, ref, task=task, lm=lm)
        x = np.array([[feat[c] for c in feat_cols]], dtype=float)
        return float(model.predict_proba(x)[0, 1])
    return predict


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "features":
        build_feature_table()
    elif cmd == "train":
        df = build_feature_table() if not FEATURES_PATH.exists() else pd.read_csv(FEATURES_PATH)
        train(df)
    elif cmd == "eval":
        df = pd.read_csv(FEATURES_PATH)
        df_lab = df.dropna(subset=["label"]).copy()
        df_lab["label"] = df_lab["label"].astype(int)
        print(f"LOSO ROC-AUC = {_loso_auc(df_lab):.3f}")
    else:
        print(f"Unknown cmd: {cmd}")
        sys.exit(1)
