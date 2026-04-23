# Plan: Robust Tongue-Tip Tracking for LROM Scoring

## Context

Current pipeline (`index.html` + `poc/build_dataset.py`) already extracts tongue tip via **HSV color thresholding** in-browser and computes `peak_auto_score` from it. MediaPipe Face Landmarker gives stable **lip/commissure landmarks** (no tongue points — feature request #5857 unresolved as of Feb 2025). Dataset is tiny: **22 webm clips × 5 subjects × 3 tasks** (`latR`, `latL`, `elev`) with ordinal Lazarus scores.

The user asked whether `dronefreak/3D-tongue-tip-tracking` (or alternative) can add real-time/offline tongue-tip prediction to improve elevation & lateralization scoring.

**Research conclusion**: `dronefreak/3D-tongue-tip-tracking` is **not a good fit** — Farnebäck optical flow + dlib CLNF, needs 2-camera checkerboard calibration, stagnant (~25 commits, no releases), no torch/ONNX path, no quantitative benchmarks. Skip it.

Target outcome: **continuous 0–100 prediction** per task (elevation, latR, latL). Lazarus bins `{0, 25, 50, 75, 100}` are for training supervision / evaluation only — final output is a real number. Offline is primary; real-time in-browser is stretch goal.

### Scoring formulas (continuous)

Let `MC = midpoint(CL, CR)` (mouth center), `IC_half = ||commissure - MC||`, `MH = ||UL - LL||`, `v = P(tongue tip visible)` from visibility classifier, `o = P(mouth open)` from landmarks.

- **Lateralization (signed)** — `lat_score = clip( (TIP.x - MC.x) / (commissure_side.x - MC.x), 0, 1 ) * 100`.
  Tip at center = 0, tip at commissure = 100, tip past commissure = 100 (clipped). Continuous.
- **Elevation** — two-branch, gated by visibility:
  - `if v < 0.3 and o > 0.5`:  `elev_score = 100` (palate contact, tip hidden behind teeth)
  - `if v < 0.3 and o < 0.5`:  **reject frame** (mouth closed — cannot assess)
  - `if v ≥ 0.3`:  `elev_score = 50 * clip( (UL.y - TIP.y) / MH + 1, 0, 2 )` → maps: tip at lower lip = 0, tip at upper lip = 50, tip above lip = 50→100 by height
- **Clip-level score** = robust percentile (e.g. 95th) of per-frame scores, not max — avoids single-frame HSV glitch.

**Resting-frame calibration**: use first ~5 frames pre-movement to record `MC_rest, CL_rest, CR_rest`. Normalize against these, not per-frame commissures — cancels facial asymmetry from stroke / Bell's palsy pulling the mouth corner during the task.

## Approach Summary (recommended)

Three-phase escalation. Stop at the earliest phase that hits clinical target (**weighted κ ≥ 0.70 vs. clinician** and **tip pixel error < 5% of inter-commissure distance**).

| Phase | What | Cost | Runs where |
|-------|------|------|-----------|
| 1. Geometric scorer | Reuse existing HSV tip + MediaPipe landmarks → compute normalized ratios → ordinal bins | ~2 days | Browser + offline Python |
| 2. SAM2 auto-labeling + YOLOv8-pose | Use SAM2 (server) to auto-label tongue tip on all frames; train 3-keypoint YOLOv8n-pose (tip + L/R commissures); export ONNX | ~1 week | Offline train, browser infer |
| 3. Hybrid image+landmark regressor | Existing `plan.md` proposal — MobileNetV3 + MLP fusion; supervised directly on clinical score | ~2 weeks | Offline |

Phases are **additive**: Phase 1 keeps serving as a fallback / consistency check in Phase 2+.

## Phase 1 — Geometric scorer + visibility classifier (do this first)

Most of the signal is **geometric ratios**, and you already have every input needed in `*_landmarks.json` + HSV tip from `index.html`. Add a **tongue-visibility classifier** so elevation=100 (tip hidden behind teeth) is distinguishable from elevation=0 (no movement) — HSV cannot tell them apart.

### Component 1: continuous geometric scorer

Formulas are listed in the Context section above. Implement in `poc/scoring.py`:
- `score_lateralization(tip, CL, CR)` → float 0–100
- `score_elevation(tip, UL, LL, visible_prob, mouth_open_prob)` → float 0–100 or `None` (reject)
- `clip_score(per_frame_scores)` → 95th percentile of valid frames.

### Component 2: visibility classifier

- Architecture: frozen MobileNetV3-small backbone + 2-layer MLP head → `P(tongue_tip_visible)`. ~1M params trainable.
- Input: 96×96 crop around MediaPipe mouth bbox.
- Labels (free, derived from existing data):
  - Positive: frames where HSV tip is inside mouth ROI with area > 50 px² and saturation > 0.4.
  - Negative: first 5 frames of each clip (pre-movement, at rest), mouth-closed frames, frames where HSV returns zero blob above threshold.
- Training: ~22 clips × ~85 frames ≈ 1800 frames = viable for a binary head.
- Output feeds elevation scorer.

### Component 3: HSV confidence score (reject weak detections)

Current HSV returns a centroid even on noise. Add `conf_hsv = f(blob_area, saturation_mean, hue_concentration)`:
- `conf_hsv < 0.3` → treat as "no tip" for scoring purposes.
- Emit confidence alongside tip in `index.html` (lines 442–519) and `build_dataset.py`.

### Peak frame / clip aggregation

Keep existing peak-frame selector for **ROI cropping**. For **scoring**, use the 95th-percentile-over-clip of per-frame scores among valid frames (visible & confident). More robust to one-frame HSV glitches.

### Files to modify

- `poc/scoring.py` (new) — formulas above.
- `poc/visibility_clf.py` (new) — train + infer binary classifier, save ONNX.
- `poc/build_dataset.py` — add columns: `tip_xy`, `conf_hsv`, `visible_prob`, `mouth_open_prob`, `lat_score`, `elev_score` per frame + per-clip aggregates.
- `poc/train.py` — baseline: `{lat_score, elev_score} → clinical_score` isotonic calibration; report MAE + quadratic-weighted κ via LOSO.
- `index.html` — add resting-frame calibration (lines 614–689), emit `conf_hsv`, call visibility ONNX (lines 442–519). Replace current `peak_auto_score` with continuous scorer.

### Expected clinical robustness gains

- **Facial asymmetry** (stroke / Bell's palsy): resting-frame calibration fixes skewed commissures.
- **Palate contact → elevation 100**: visibility classifier catches "tip not visible + mouth open", which HSV-alone misses.
- **Low-amplitude movement** (ALS, Parkinson): 95th-percentile over clip + continuous score captures subclinical amplitude better than discrete bins.
- **Pale/coated tongue**: `conf_hsv` drops → falls back to visibility classifier only (less accurate tip location but at least correct category).
- **Mouth closed**: explicit rejection instead of garbage score.

### Why this first

- 22 clips is way too few to train anything deep. The geometric baseline establishes a performance floor and exposes which failure modes matter (bad HSV detection? head pose? occlusion?).
- If weighted κ on 22 clips hits ≥ 0.70, phases 2–3 are justified only by scaling, not by accuracy.
- Zero new dependencies.

## Phase 2 — SAM2 auto-labeling + YOLOv8-pose

Triggers **only if** Phase 1 geometric scorer fails clinical target, OR HSV tip is unreliable under lighting variation / facial hair / lipstick.

### 2a. Label with SAM2 (offline, GPU required)

- Install `facebookresearch/sam2` (Apache-2.0). Use SAM 2.1 Hiera-Small (~39M params).
- For each clip: click tongue in first frame of peak region → SAM2 propagates mask across all ~85 frames (~44 fps on one A100; CPU-slow but feasible for 22 clips).
- Extract tip = lowest pixel of mask (elevation) or rightmost/leftmost (latR/latL). Much more robust than HSV under shadows/reflection.
- Dump per-frame `{tip_xy, commissures_xy}` to JSON → supervision for YOLOv8.

Reference: [SAM2 paper](https://arxiv.org/html/2408.00714), [repo](https://github.com/facebookresearch/sam2).

Alternative if no GPU: **TongueSAM** (cshan-github/TongueSAM) — zero-shot tongue mask, lighter.

### 2b. Train YOLOv8n-pose (3 keypoints)

- Ultralytics YOLOv8n-pose, AGPL (or switch to RTMPose-s Apache-2.0 if license matters).
- Keypoints: `[tongue_tip, commissure_L, commissure_R]` with visibility flag.
- Training set: ~22 clips × ~85 frames = ~1800 frames. Augment heavy (flip swaps L↔R + `latR`↔`latL` label), color jitter, random erasing.
- LOSO split across 5 subjects.
- Export ONNX → run in browser via **onnxruntime-web** (WebGPU backend <30 ms/frame on modern laptops).

### 2c. Scoring layer on top of keypoints — ensemble with Phase 1

Same formulas as Phase 1, but with **weighted ensemble** of YOLO + HSV tips:

```
tip = (conf_yolo * tip_yolo + conf_hsv * tip_hsv) / (conf_yolo + conf_hsv)
if |tip_yolo - tip_hsv| > 0.10 * IC → flag_low_agreement = True (show both to clinician)
```

Commissures from YOLO (more robust under asymmetry) with MediaPipe as fallback.

YOLO provides a trained `objectness` score per keypoint — acts as its own confidence. This gives the model a real "no tongue visible" output, complementing the visibility classifier.

### Files to add

- `poc/label_with_sam2.py` — batch SAM2 annotator.
- `poc/yolo_pose_train.py` — Ultralytics wrapper; writes `runs/pose/...`.
- `poc/yolo_pose_export.py` — ONNX export + INT8 quant option.
- `poc/ort_web_infer.js` — browser-side ONNX runtime glue; hook into `index.html` replacing HSV block (lines 442–519).

## Phase 3 — End-to-end regressor (existing `plan.md`)

Only if keypoint-based scoring still fails. Already specced in `/home/lucas_takanori/oro_lat_el/plan.md`:
hybrid MobileNetV3-small image branch + MLP landmark branch → 1 scalar, MAE + CORAL loss, LOSO CV.
Feed YOLO keypoints from Phase 2 into the MLP branch (stronger than HSV-only tip).

Do **not** skip to Phase 3 — supervised-on-22-clips deep regressors will overfit badly.

## Critical files referenced

- `/home/lucas_takanori/oro_lat_el/index.html` — webcam pipeline, HSV tip detection lines 442–519, MediaPipe init lines 380–395, score bins lines 338–346.
- `/home/lucas_takanori/oro_lat_el/poc/build_dataset.py` — ROI crop logic, peak-frame scorer, feature extraction, CSV writer.
- `/home/lucas_takanori/oro_lat_el/poc/render_overlays.py` — QA visualization (reuse for Phase 2 label QC).
- `/home/lucas_takanori/oro_lat_el/poc/train.py` — LOSO Ridge/RF baseline (extend for geometric scorer eval).
- `/home/lucas_takanori/oro_lat_el/poc/out/crops/dataset.csv` — current manifest (22 rows).
- `/home/lucas_takanori/oro_lat_el/data/*/` — raw webm + `_landmarks.json` + `_meta.json`.
- `/home/lucas_takanori/oro_lat_el/plan.md` — previous architecture proposal (Phase 3 origin).

## Verification

### Phase 1
- Run `python poc/build_dataset.py` → confirm new `geo_score_latR/latL/elev` columns.
- Run `python poc/train.py --mode=geometric` → print per-task MAE, accuracy on discretized bins, quadratic-weighted kappa across LOSO folds.
- Visual QC: overlay predicted score vs. clinical on the 9 existing overlay videos in `poc/overlays/Lucas/`.

### Phase 2
- SAM2 mask overlay video per clip → manual inspection (5 min sanity check).
- YOLO val mAP@0.5 on held-out subject ≥ 0.80 for tip keypoint.
- Browser E2E: open `index.html`, perform all 3 tasks, confirm ONNX runtime returns tip within ~3 px of HSV tip on clean lighting AND recovers when HSV fails (test by dimming room).

### Phase 3
- Existing `plan.md` eval protocol.

## Data volume — blocker for Phase 2/3

22 clips / 5 subjects is **below** viable for a 3-keypoint pose model even with SAM2 labels. Before committing to Phase 2, confirm ability to collect **≥ 30 subjects × 3 tasks × 3 levels ≈ 270 clips** via current `index.html` collection app. Otherwise stop at Phase 1.

## Non-goals

- No 3D reconstruction (steliosploumpis/tongue, single-image 3D, `dronefreak`). Single-view 2D is sufficient for Lazarus ordinal scoring.
- No full tongue segmentation at inference — keypoints are enough, and SAM2 is too heavy for browser.
- No head-pose normalization module until Phase 1 shows pose is a failure mode.
