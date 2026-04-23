"""Continuous geometric scoring for LROM (Phase 1).

Pipeline per frame:
  HSV tongue-tip detection (with confidence) + MediaPipe lip landmarks
    → lateralization score (continuous 0-100, signed by task side)
    → elevation score (continuous 0-100, gated by visibility)

Clip-level aggregation: 95th-percentile over valid frames.

Resting-frame calibration: uses first N frames (pre-movement) to fix
commissure reference — cancels facial asymmetry (stroke/Bell's palsy).

Run:  python -m poc.scoring  (smoke-test on first data clip)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

LM = {
    "R_COMMISSURE": 61,
    "L_COMMISSURE": 291,
    "UPPER_LIP_CENTER": 13,
    "LOWER_LIP_CENTER": 14,
    "LIP_LEFT_OUTER": 78,
    "LIP_RIGHT_OUTER": 308,
    "NOSE_TIP": 1,
    "CHIN": 152,
}

# MediaPipe inner-lip ring (closed polygon, clockwise):
# upper arch (right→left): 78 → 191 → 80 → 81 → 82 → 13 → 312 → 311 → 310 → 415 → 308
# lower arch (left→right): 308 → 324 → 318 → 402 → 317 → 14 → 87 → 178 → 88 → 95 → 78
INNER_LIP_RING = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
]

HUE_MIN_A, HUE_MAX_A = 330.0, 360.0
HUE_MIN_B, HUE_MAX_B = 0.0, 20.0
SAT_MIN = 0.25
VAL_MIN, VAL_MAX = 0.35, 0.90

REST_FRAMES = 5
MIN_BLOB_AREA_FRAC = 0.002
GOOD_BLOB_AREA_FRAC = 0.03


@dataclass
class RefGeometry:
    """Per-frame or resting reference geometry, in pixels."""
    CL: np.ndarray     # right-commissure (subject's right = image left when not mirrored)
    CR: np.ndarray     # left-commissure
    UL: np.ndarray     # upper-lip center
    LL: np.ndarray     # lower-lip center
    MC: np.ndarray     # mouth center = midpoint(CL, CR)
    IC: float          # inter-commissure width
    MH: float          # mouth opening height


@dataclass
class TipDetection:
    xy: Optional[np.ndarray]   # tip px coords, or None
    conf: float                # 0..1 confidence
    blob_area_frac: float      # blob area / mouth ROI area
    sat_mean: float            # mean saturation of blob
    in_mouth: bool             # blob overlaps mouth bbox


def ref_from_landmarks(lm, H: int, W: int) -> RefGeometry:
    def p(i):
        return np.array([lm[i]["x"] * W, lm[i]["y"] * H], dtype=np.float64)
    CL = p(LM["R_COMMISSURE"])
    CR = p(LM["L_COMMISSURE"])
    UL = p(LM["UPPER_LIP_CENTER"])
    LL = p(LM["LOWER_LIP_CENTER"])
    MC = 0.5 * (CL + CR)
    IC = float(np.linalg.norm(CR - CL))
    MH = float(np.linalg.norm(UL - LL))
    return RefGeometry(CL=CL, CR=CR, UL=UL, LL=LL, MC=MC, IC=IC, MH=MH)


def resting_reference(landmark_frames: list, H: int, W: int,
                      n_frames: int = REST_FRAMES) -> RefGeometry:
    """Average geometry over first n frames (assumed pre-movement).

    Use these points instead of per-frame landmarks so tongue-movement
    pull on the commissures doesn't contaminate the denominator. Also
    reduces sensitivity to per-frame MediaPipe jitter and to resting
    facial asymmetry (stroke / Bell's palsy).
    """
    k = min(n_frames, len(landmark_frames))
    CLs, CRs, ULs, LLs = [], [], [], []
    for i in range(k):
        r = ref_from_landmarks(landmark_frames[i]["lm"], H, W)
        CLs.append(r.CL); CRs.append(r.CR); ULs.append(r.UL); LLs.append(r.LL)
    CL = np.mean(CLs, axis=0)
    CR = np.mean(CRs, axis=0)
    UL = np.mean(ULs, axis=0)
    LL = np.mean(LLs, axis=0)
    MC = 0.5 * (CL + CR)
    IC = float(np.linalg.norm(CR - CL))
    MH = float(np.linalg.norm(UL - LL))
    return RefGeometry(CL=CL, CR=CR, UL=UL, LL=LL, MC=MC, IC=IC, MH=MH)


def _inner_lip_polygon(lm, H: int, W: int) -> np.ndarray:
    """MediaPipe inner-lip ring as Nx2 int array of (x, y) pixel points."""
    pts = np.array([[lm[i]["x"] * W, lm[i]["y"] * H] for i in INNER_LIP_RING],
                   dtype=np.float32)
    return pts


def _polygon_mask(poly_xy: np.ndarray, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_xy.astype(np.int32)], 1)
    return mask


def _extend_polygon_horizontally(poly_xy: np.ndarray, ref: RefGeometry,
                                 pad_frac: float = 0.30) -> np.ndarray:
    """Push leftmost points further left and rightmost further right by pad·IC.

    Used for lat tasks: tongue tip can protrude laterally past the inner-lip
    polygon. Preserves top/bottom to stay out of beard.
    """
    poly = poly_xy.copy()
    pad = ref.IC * pad_frac
    mid_x = ref.MC[0]
    # Expand only the two extreme commissure-area points (kept thin vertically).
    left_mask = poly[:, 0] < mid_x
    right_mask = ~left_mask
    # Bias the leftmost 25% leftward and rightmost 25% rightward by pad.
    # Use hard threshold on x distance from centre.
    left_thresh = np.percentile(poly[left_mask, 0], 25) if left_mask.any() else mid_x
    right_thresh = np.percentile(poly[right_mask, 0], 75) if right_mask.any() else mid_x
    poly[poly[:, 0] <= left_thresh, 0] -= pad
    poly[poly[:, 0] >= right_thresh, 0] += pad
    return poly


def _mouth_mask_and_bbox(ref: RefGeometry, H: int, W: int,
                         task: str = "elev",
                         lm=None,
                         ) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Return (bbox=(x0,y0,x1,y1), binary ROI mask in full-frame coords).

    When `lm` is provided, uses the MediaPipe inner-lip polygon as the ROI,
    precisely excluding beard/outer-lip pixels that a rectangular bbox picks
    up. For lat tasks, the polygon is extended horizontally past commissure
    to catch protruding tongue tip. Falls back to a rectangular bbox when
    `lm` is None (called by legacy code paths).
    """
    if lm is not None:
        poly = _inner_lip_polygon(lm, H, W)
        if task.startswith("lat"):
            poly = _extend_polygon_horizontally(poly, ref, pad_frac=0.60)
        if task == "elev":
            # Allow tongue to protrude above upper-lip ring for elevation.
            up_pad = 0.35 * max(ref.MH, 1.0)
            poly = poly.copy()
            poly[poly[:, 1] < ref.MC[1], 1] -= up_pad
        mask = _polygon_mask(poly, H, W)
        xs = poly[:, 0]
        ys = poly[:, 1]
        x0, y0 = int(max(0, xs.min())), int(max(0, ys.min()))
        x1, y1 = int(min(W, xs.max())), int(min(H, ys.max()))
        if x1 - x0 < 4 or y1 - y0 < 4:
            mask = np.ones((H, W), dtype=np.uint8)
            x0, y0, x1, y1 = 0, 0, W, H
        return (x0, y0, x1, y1), mask

    # --- fallback: rectangular bbox ---
    MH = max(ref.MH, 1.0)
    if task.startswith("lat"):
        x0 = int(max(0, min(ref.CL[0], ref.CR[0]) - ref.IC * 0.30))
        x1 = int(min(W, max(ref.CL[0], ref.CR[0]) + ref.IC * 0.30))
        y0 = int(max(0, ref.UL[1] + 0.10 * MH))
        y1 = int(min(H, ref.LL[1] - 0.10 * MH))
    else:
        x0 = int(max(0, min(ref.CL[0], ref.CR[0]) - ref.IC * 0.10))
        x1 = int(min(W, max(ref.CL[0], ref.CR[0]) + ref.IC * 0.10))
        y0 = int(max(0, ref.UL[1] - 0.60 * MH))
        y1 = int(min(H, ref.LL[1] + 0.25 * MH))
    if x1 - x0 < 4 or y1 - y0 < 4:
        x0, y0, x1, y1 = 0, 0, W, H
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return (x0, y0, x1, y1), mask


CR_MIN = 150          # YCrCb Cr channel min — tongue has higher red-chroma than skin
RED_EXCESS_MIN = 35   # 2R − G − B threshold (bgr) — filters dull-red skin/beard


def _hsv_tongue_mask(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (binary tongue mask, saturation 0..1, value 0..1).

    Uses YCrCb Cr channel (red chroma) + red-excess test to separate
    tongue (saturated red) from skin/beard (dull reddish). HSV saturation
    is still returned (needed by visibility classifier features).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    Cr = ycc[..., 1].astype(np.float32)

    b = frame_bgr[..., 0].astype(np.float32)
    g = frame_bgr[..., 1].astype(np.float32)
    r = frame_bgr[..., 2].astype(np.float32)
    red_excess = 2 * r - g - b

    tongue = (Cr >= CR_MIN) & (red_excess >= RED_EXCESS_MIN) \
        & (V >= VAL_MIN) & (V <= VAL_MAX) & (S >= SAT_MIN)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tongue_u8 = tongue.astype(np.uint8)
    tongue_u8 = cv2.morphologyEx(tongue_u8, cv2.MORPH_OPEN, kernel)
    return tongue_u8, S, V


def detect_tongue_tip(frame_bgr: np.ndarray, ref: RefGeometry, task: str,
                      mirrored: bool = False, lm=None) -> TipDetection:
    """HSV-based tongue-tip detection with confidence score.

    task ∈ {"latR","latL","elev"}. For lat tasks, tip = extremum in
    the movement direction. For elev, tip = topmost (smallest y) point.
    "mirrored" flips L/R when webcam is mirrored.
    "lm" (per-frame MediaPipe landmarks) enables inner-lip polygon ROI.
    """
    H, W = frame_bgr.shape[:2]
    (x0, y0, x1, y1), mouth_full = _mouth_mask_and_bbox(ref, H, W, task=task, lm=lm)
    tongue_full, S, _V = _hsv_tongue_mask(frame_bgr)
    tongue = tongue_full & mouth_full

    if tongue.sum() == 0:
        return TipDetection(xy=None, conf=0.0, blob_area_frac=0.0,
                            sat_mean=0.0, in_mouth=False)

    # Largest connected component inside mouth ROI.
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(tongue, 8)
    if num <= 1:
        return TipDetection(xy=None, conf=0.0, blob_area_frac=0.0,
                            sat_mean=0.0, in_mouth=False)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    blob = (labels == best)
    ys, xs = np.where(blob)
    if xs.size == 0:
        return TipDetection(xy=None, conf=0.0, blob_area_frac=0.0,
                            sat_mean=0.0, in_mouth=False)

    mouth_area = float((x1 - x0) * (y1 - y0)) + 1e-9
    blob_area_frac = float(blob.sum()) / mouth_area
    sat_mean = float(S[blob].mean())

    # Tip selection per task. Use a robust percentile (90%) in the
    # target direction, not absolute extremum — a single stray red pixel
    # at the mouth edge would otherwise dominate the score.
    # Axis: x grows right in image. Non-mirrored: subject-right = image-left.
    # Mirrored (selfie): subject-right = image-right.
    if task == "latR":
        tip_x = float(np.percentile(xs, 90 if mirrored else 10))
    elif task == "latL":
        tip_x = float(np.percentile(xs, 10 if mirrored else 90))
    else:  # elev → topmost (smallest y)
        tip_x = float(np.percentile(xs, 50))
    if task == "elev":
        tip_y = float(np.percentile(ys, 10))
    else:
        # Median y among pixels near the target-direction extremum.
        if task == "latR":
            thresh = np.percentile(xs, 85 if mirrored else 15)
            mask_dir = (xs >= thresh) if mirrored else (xs <= thresh)
        else:  # latL
            thresh = np.percentile(xs, 15 if mirrored else 85)
            mask_dir = (xs <= thresh) if mirrored else (xs >= thresh)
        tip_y = float(np.median(ys[mask_dir])) if mask_dir.any() else float(np.median(ys))
    tip = np.array([tip_x, tip_y])

    # Confidence: plausibility of detection.
    #   area: below MIN_BLOB_AREA_FRAC → 0; ramps to 1 at GOOD_BLOB_AREA_FRAC.
    #   saturation: below SAT_MIN → 0; ramps to 1 at 0.6.
    area_term = np.clip(
        (blob_area_frac - MIN_BLOB_AREA_FRAC)
        / max(GOOD_BLOB_AREA_FRAC - MIN_BLOB_AREA_FRAC, 1e-6), 0.0, 1.0)
    sat_term = np.clip((sat_mean - SAT_MIN) / (0.6 - SAT_MIN), 0.0, 1.0)
    conf = float(area_term * sat_term)

    return TipDetection(xy=tip, conf=conf, blob_area_frac=blob_area_frac,
                        sat_mean=sat_mean, in_mouth=True)


def mouth_open_prob(ref: RefGeometry) -> float:
    """Proxy P(mouth open): mouth-aspect ratio, clipped 0..1."""
    aspect = ref.MH / max(ref.IC, 1e-6)
    return float(np.clip((aspect - 0.05) / (0.35 - 0.05), 0.0, 1.0))


def score_lateralization(tip_xy: np.ndarray, ref: RefGeometry,
                         task: str, mirrored: bool = False) -> float:
    """Continuous 0..100. Signed by expected task direction.

    0  = tip at mouth center (or moved opposite direction)
    100 = tip at or past the target-side commissure.
    """
    # Expected in-image direction for the tongue tip.
    # Non-mirrored: subject-right = image-left (smaller x).
    # Mirrored (selfie): subject-right = image-right (larger x).
    if task == "latR":
        sign = +1 if mirrored else -1
    else:  # latL
        sign = -1 if mirrored else +1
    signed_disp = sign * (tip_xy[0] - ref.MC[0])
    half_ic = ref.IC / 2.0
    if half_ic < 1e-6:
        return 0.0
    ratio = signed_disp / half_ic
    return float(np.clip(ratio, 0.0, 1.0) * 100.0)


def score_elevation(tip: TipDetection, ref: RefGeometry,
                    visible_prob: Optional[float] = None,
                    vis_threshold: float = 0.5,
                    open_threshold: float = 0.5) -> Optional[float]:
    """Continuous 0..100, or None if unassessable (mouth closed).

    Two-branch gated by visibility:
      not visible + mouth open  → 100 (palate contact, tip hidden)
      not visible + mouth closed→ None (cannot assess)
      visible → linear interp on tip vertical position relative to lip span
    """
    o = mouth_open_prob(ref)
    # If caller didn't pass a classifier prob, derive from tip detection conf.
    v = visible_prob if visible_prob is not None else tip.conf

    if v < vis_threshold:
        if o < open_threshold:
            return None                # mouth closed → cannot score
        return 100.0                   # mouth open, no tip visible → palate

    if tip.xy is None or ref.MH < 1e-6:
        return None

    # Clinical rubric: visible tip ⇒ elevation ≤ 50. Only palate contact
    # (tip hidden, handled above) reaches 100. Linear in vertical position:
    #   tip at lower lip (y = LL.y) → 0
    #   tip at upper lip (y = UL.y) → 50
    #   tip above upper lip, visible → stays at 50 (close-to-palate but not hidden)
    # y grows downward in images.
    delta = (ref.LL[1] - tip.xy[1]) / ref.MH   # 0 at lower, 1 at upper, >1 above
    return float(np.clip(delta * 50.0, 0.0, 50.0))


def aggregate_clip(per_frame_scores: list, percentile: float = 95.0) -> Optional[float]:
    """Robust clip-level score: percentile over valid (non-None) frames."""
    valid = [s for s in per_frame_scores if s is not None and np.isfinite(s)]
    if not valid:
        return None
    return float(np.percentile(valid, percentile))


def tip_from_mask(mask: np.ndarray, task: str,
                  mirrored: bool = False,
                  roi_mask: np.ndarray = None) -> TipDetection:
    """Derive tongue tip from a SAM2 binary mask.

    tip = task-direction extremum of mask pixels (percentile 95/5 for robustness).
    If roi_mask is provided, mask is intersected with it first.
    """
    m = mask.astype(bool)
    if roi_mask is not None:
        m = m & roi_mask.astype(bool)
    if not m.any():
        return TipDetection(xy=None, conf=0.0, blob_area_frac=0.0,
                            sat_mean=0.0, in_mouth=False)
    ys, xs = np.where(m)
    # Task-direction tip extremum (robust percentile)
    if task == "latR":
        tip_x = float(np.percentile(xs, 95 if mirrored else 5))
        thresh = np.percentile(xs, 90 if mirrored else 10)
        mask_dir = (xs >= thresh) if mirrored else (xs <= thresh)
    elif task == "latL":
        tip_x = float(np.percentile(xs, 5 if mirrored else 95))
        thresh = np.percentile(xs, 10 if mirrored else 90)
        mask_dir = (xs <= thresh) if mirrored else (xs >= thresh)
    else:  # elev → topmost
        tip_x = float(np.percentile(xs, 50))
        tip_y_pct = float(np.percentile(ys, 5))
        area = float(m.sum())
        H, W = m.shape
        area_frac = area / float(H * W)
        conf = float(np.clip(area_frac * 200.0, 0.0, 1.0))   # area-only conf
        return TipDetection(xy=np.array([tip_x, tip_y_pct]),
                            conf=conf, blob_area_frac=area_frac,
                            sat_mean=0.0, in_mouth=True)
    tip_y = float(np.median(ys[mask_dir])) if mask_dir.any() else float(np.median(ys))
    H, W = m.shape
    area_frac = float(m.sum()) / float(H * W)
    conf = float(np.clip(area_frac * 200.0, 0.0, 1.0))
    return TipDetection(xy=np.array([tip_x, tip_y]), conf=conf,
                        blob_area_frac=area_frac, sat_mean=0.0, in_mouth=True)


def score_clip_with_masks(video_frames: list, landmark_frames: list,
                           masks: np.ndarray, task: str,
                           mirrored: bool = False) -> dict:
    """Score a clip using precomputed SAM2 masks for tongue segmentation.

    masks: bool array (n_frames, H, W). Tip is derived as the task-direction
    extremum of each mask intersected with the (lateral-extended) inner-lip
    polygon ROI so that pixels outside the mouth are ignored.
    """
    n = min(len(video_frames), len(landmark_frames), len(masks))
    if n == 0:
        return {"lat_score": None, "elev_score": None, "n_valid": 0,
                "n_frames": 0, "mean_tip_conf": 0.0, "mean_vis_prob": 0.0}
    H, W = video_frames[0].shape[:2]
    rest = resting_reference(landmark_frames, H, W)

    lat_scores, elev_scores, confs = [], [], []
    vis_probs = []
    for i in range(n):
        lm = landmark_frames[i]["lm"]
        ref_f = ref_from_landmarks(lm, H, W)
        ref_eff = RefGeometry(
            CL=rest.CL, CR=rest.CR,
            UL=ref_f.UL, LL=ref_f.LL, MC=rest.MC,
            IC=rest.IC, MH=float(np.linalg.norm(ref_f.UL - ref_f.LL)),
        )
        (_, _, _, _), roi_mask = _mouth_mask_and_bbox(ref_eff, H, W,
                                                      task=task, lm=lm)
        tip = tip_from_mask(masks[i], task, mirrored=mirrored,
                            roi_mask=roi_mask)
        confs.append(tip.conf)
        # Visibility derived from mask area: higher area within ROI = visible.
        vis = tip.conf
        vis_probs.append(vis)

        if task.startswith("lat") and tip.xy is not None and tip.conf > 0.05:
            lat_scores.append(
                score_lateralization(tip.xy, ref_eff, task, mirrored=mirrored))
        elif task == "elev":
            es = score_elevation(tip, ref_eff, visible_prob=vis,
                                  vis_threshold=0.15)
            if es is not None:
                elev_scores.append(es)

    return {
        "lat_score": aggregate_clip(lat_scores, percentile=80.0)
                      if task.startswith("lat") else None,
        "elev_score": (float(np.mean(elev_scores)) if elev_scores else None)
                       if task == "elev" else None,
        "n_valid": len(lat_scores if task.startswith("lat") else elev_scores),
        "n_frames": n,
        "mean_tip_conf": float(np.mean(confs)) if confs else 0.0,
        "mean_vis_prob": float(np.mean(vis_probs)) if vis_probs else 0.0,
    }


def score_clip(video_frames: list, landmark_frames: list, task: str,
               mirrored: bool = False,
               visibility_fn=None) -> dict:
    """Full per-clip scoring. Returns dict with continuous scores + diagnostics.

    visibility_fn: optional callable(frame_bgr, ref, tip) -> float in [0,1].
                   If None, tip.conf is used as proxy for visibility.
    """
    n = min(len(video_frames), len(landmark_frames))
    if n == 0:
        return {"lat_score": None, "elev_score": None, "n_valid": 0}

    H, W = video_frames[0].shape[:2]
    rest_ref = resting_reference(landmark_frames, H, W)

    lat_scores, elev_scores, tip_confs, vis_probs = [], [], [], []
    tips_xy = []
    for i in range(n):
        lm = landmark_frames[i]["lm"]
        ref_frame = ref_from_landmarks(lm, H, W)
        # Use resting commissures (stable reference), but per-frame lip centers
        # (tongue can push the lower lip open — we want that dynamic info).
        ref_eff = RefGeometry(
            CL=rest_ref.CL, CR=rest_ref.CR,
            UL=ref_frame.UL, LL=ref_frame.LL, MC=rest_ref.MC,
            IC=rest_ref.IC, MH=float(np.linalg.norm(ref_frame.UL - ref_frame.LL)),
        )
        tip = detect_tongue_tip(video_frames[i], ref_eff, task,
                                mirrored=mirrored, lm=lm)
        tip_confs.append(tip.conf)
        tips_xy.append(tip.xy.tolist() if tip.xy is not None else None)

        v = (visibility_fn(video_frames[i], ref_eff, tip, task=task, lm=lm)
             if visibility_fn else tip.conf)
        vis_probs.append(float(v))

        if task.startswith("lat") and tip.xy is not None and tip.conf > 0.15:
            # Lat scoring uses HSV confidence only — the trained visibility
            # classifier is biased toward elevation-style round blobs and
            # down-weights the thin lateral-tongue shape.
            lat_scores.append(
                score_lateralization(tip.xy, ref_eff, task, mirrored=mirrored))
        elif task == "elev":
            es = score_elevation(tip, ref_eff, visible_prob=v)
            if es is not None:
                elev_scores.append(es)

    # Lat: 80th-pct (peak reach, robust to single-pixel outliers).
    # Elev: mean (palate contact should be sustained, not peak).
    out = {
        "lat_score": aggregate_clip(lat_scores, percentile=80.0) if task.startswith("lat") else None,
        "elev_score": (float(np.mean(elev_scores)) if elev_scores else None) if task == "elev" else None,
        "n_valid": len(lat_scores if task.startswith("lat") else elev_scores),
        "n_frames": n,
        "mean_tip_conf": float(np.mean(tip_confs)) if tip_confs else 0.0,
        "mean_vis_prob": float(np.mean(vis_probs)) if vis_probs else 0.0,
        "tips_xy": tips_xy,
        "rest_ref": {
            "CL": rest_ref.CL.tolist(), "CR": rest_ref.CR.tolist(),
            "UL": rest_ref.UL.tolist(), "LL": rest_ref.LL.tolist(),
            "IC": rest_ref.IC, "MH": rest_ref.MH,
        },
    }
    return out


# -----------------------------------------------------------------------------
# Smoke test: run on a single clip from data/ when executed directly.
# -----------------------------------------------------------------------------
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


if __name__ == "__main__":
    import json
    import sys

    root = Path(__file__).resolve().parent.parent
    data = root / "data"
    # Pick first available clip.
    for subj in sorted(p for p in data.iterdir() if p.is_dir()):
        metas = sorted(subj.glob("*_meta.json"))
        if not metas:
            continue
        m = metas[0]
        stem = m.name.replace("_meta.json", "")
        vid = subj / f"{stem}.webm"
        lmp = subj / f"{stem}_landmarks.json"
        if not (vid.exists() and lmp.exists()):
            continue
        task = stem.split("_")[0]
        meta = json.loads(m.read_text())
        lm_frames = json.loads(lmp.read_text())["landmarks"]
        vf = _load_video(vid)
        res = score_clip(vf, lm_frames, task, mirrored=bool(meta.get("cameraFlipped")))
        print(f"[{subj.name}/{stem}] task={task}")
        print(f"  lat_score  = {res['lat_score']}")
        print(f"  elev_score = {res['elev_score']}")
        print(f"  n_valid    = {res['n_valid']} / {res['n_frames']}")
        print(f"  mean tip conf = {res['mean_tip_conf']:.3f}")
        sys.exit(0)
